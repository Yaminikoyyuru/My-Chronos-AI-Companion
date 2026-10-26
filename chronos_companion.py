"""Chronos Care - agentic vision companion for elders living alone.

Perception (OpenCV)  ->  Decision (escalation ladder)  ->  Action (voice, AWS S3, doctor approval)
Run:  python chronos_companion.py                 (real timers: 30 s still, 15 s face lost)
      python chronos_companion.py --demo          (short timers for the video: 8 s / 6 s)
      python chronos_companion.py --immobile 5 --facelost 5    (custom timers)
"""
import argparse
import json
import os
import random
import threading
import time
from datetime import datetime

import cv2
import numpy as np

# Audio and .env support are optional, so the app also runs on a judge's machine
# without microphone / speakers / PortAudio (use hotkeys y and h instead of speaking).
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
try:
    import sounddevice as sd
    import speech_recognition as sr
except Exception:
    sd = sr = None
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

from cloud import CloudClient  # noqa: E402
from perception import PerceptionPipeline, build_face_detector, BASE_DIR  # noqa: E402
from robot_ui import draw_full_robot  # noqa: E402

SAMPLE_RATE = 16000
MOVEMENT_RESPONSE = 0.05  # motion fraction that counts as "patient is moving"
MOVEMENT_FRAMES = 15      # ...and it must last this many frames (about 0.5 s)


# ----------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------
class State:
    robot_state = "IDLE"
    message = "Listening for your voice..."
    is_talking = False
    listening = False
    incident_active = False
    latest_frame = None
    peak_motion = 0.0
    motion_frames = 0
    last_voice = ""
    paused = False        # patient privacy break / sleep
    pause_mode = None     # "break" or "sleep"
    pause_until = 0.0     # auto-resume time (safety: monitoring never stays off forever)


st = State()
patient = {
    "gender": "Unknown",
    "age": "Unknown",
    "condition": "General Wellness",
    "language": "en-IN",  # English only
    "sender": os.getenv("SENDER_EMAIL", ""),
    "receiver": os.getenv("RECEIVER_EMAIL", ""),
}
cloud = None
pipeline = None
cfg = {"doctor_wait": 180, "break_s": 30 * 60, "sleep_s": 8 * 3600}
mic_lock = threading.Lock()
quiz_state = {"answer": None}
prompt_tracker = {"morning_walk": False, "breakfast": False, "songs": False}
last_voice_time = 0.0

QUIZ_QUESTIONS = [
    ("Which animal is called the king of the jungle?", {"lion"}),
    ("How many days are there in a week?", {"seven", "7"}),
    ("Which festival is known as the festival of lights?", {"diwali", "deepavali"}),
    ("What color do you get by mixing blue and yellow?", {"green"}),
]

# Voice keywords (English; a few Telugu words written in English letters are kept).
DISTRESS = {"baledhu", "baledu", "bagaledu", "pain", "help", "emergency", "not well"}
OK = {"fine", "good", "okay", "ok", "yes", "bagunnanu", "alright"}
PAUSE = {"pause", "break", "washroom", "bathroom", "toilet"}
SLEEP = {"sleep", "nap", "good night"}
RESUME = {"back", "i am back", "resume", "awake"}
QUIZ_WORDS = {"quiz", "game", "question"}
NEWS_WORDS = {"news", "headlines"}
SONG_WORDS = {"song", "songs", "music"}
HELLO_WORDS = {"hello", "hi"}

PROFILE = {
    "FALL_SUSPECTED": dict(wait=6, accept_motion=False,
                           msg="I think you may have fallen. Are you hurt? Please say I am fine, or say help."),
    "IMMOBILE": dict(wait=8, accept_motion=True,
                     msg="I noticed you have not moved for a while. Are you okay? Please say I am fine."),
    "FACE_LOST": dict(wait=10, accept_motion=False,
                      msg="I cannot see you anymore. Are you okay? Please say something."),
    "VOICE_DISTRESS": dict(wait=0, accept_motion=False, msg=""),
}


def has_any(phrase, keywords):
    """True if any keyword is in the phrase (multi-word keywords use substring match)."""
    tokens = set(phrase.split())
    for k in keywords:
        if " " in k or not k.isascii():
            if k in phrase:
                return True
        elif k in tokens:
            return True
    return False


# ----------------------------------------------------------------------
# Trace (proof of perception -> decision -> action for judges)
# ----------------------------------------------------------------------
def trace(event, **data):
    record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event}
    record.update(data)
    print("[TRACE]", record)
    try:
        with open(os.path.join(BASE_DIR, "agent_trace.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    cloudwatch_log(record)   # CloudWatch Logs (silently skipped if unavailable)


# ----------------------------------------------------------------------
# CloudWatch Logs (optional). If the account blocks it, it switches itself off quietly.
# ----------------------------------------------------------------------
_cw = {"client": None, "stream": None, "ok": True, "lock": threading.Lock()}


def cloudwatch_log(record):
    if not _cw["ok"] or os.getenv("CLOUDWATCH", "on").lower() == "off":
        return
    if cloud is not None and getattr(cloud, "local_mode", False):
        return
    with _cw["lock"]:
        try:
            import boto3
            if _cw["client"] is None:
                _cw["client"] = boto3.client("logs", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            logs = _cw["client"]
            group = os.getenv("CW_LOG_GROUP", "/chronos/agent")
            if _cw["stream"] is None:
                try:
                    logs.create_log_group(logGroupName=group)
                except Exception as e:
                    if "ResourceAlreadyExists" not in str(e):
                        raise
                _cw["stream"] = "run-" + datetime.now().strftime("%Y%m%dT%H%M%S")
                logs.create_log_stream(logGroupName=group, logStreamName=_cw["stream"])
            logs.put_log_events(
                logGroupName=group, logStreamName=_cw["stream"],
                logEvents=[{"timestamp": int(time.time() * 1000),
                            "message": json.dumps(record, ensure_ascii=False)}])
        except Exception as e:
            print("[CloudWatch] logging disabled:", str(e)[:150])
            _cw["ok"] = False


# ----------------------------------------------------------------------
# Speech in / out
# ----------------------------------------------------------------------
def robot_speak(text):
    st.message = text

    def work():
        try:
            st.is_talking = True
            engine = pyttsx3.init()
            engine.setProperty("rate", 140)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            st.is_talking = False

    threading.Thread(target=work, daemon=True).start()


def wait_quiet():
    time.sleep(0.4)
    while st.is_talking:
        time.sleep(0.2)


injected = {"phrase": ""}   # phrase typed with a hotkey (y = "i am fine", h = "help")


def take_injected():
    phrase, injected["phrase"] = injected["phrase"], ""
    return phrase


def listen_phrase(seconds=4, lang=None, allow_inject=False):
    """Record from the mic and return lowercase text. Hotkey phrases (y/h) are only used when
    allow_inject=True, i.e. while the agent is waiting for an answer during an incident."""
    lang = lang or patient["language"]
    if sd is None or sr is None:          # no audio stack: just wait and accept hotkey phrases
        end = time.time() + seconds
        while time.time() < end:
            if allow_inject and injected["phrase"]:
                return take_injected()
            time.sleep(0.1)
        return ""
    text = ""
    with mic_lock:
        try:
            data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            sd.wait()
            audio = sr.AudioData(data.tobytes(), SAMPLE_RATE, 2)
            text = sr.Recognizer().recognize_google(audio, language=lang).lower().strip()
        except Exception:
            text = ""
    return (take_injected() if allow_inject else "") or text


def is_distress(phrase):
    return has_any(phrase, DISTRESS)


def is_ok(phrase):
    return has_any(phrase, OK) and "not" not in set(phrase.split()) and not is_distress(phrase)


# ----------------------------------------------------------------------
# Agentic incident ladder
# ----------------------------------------------------------------------
def check_in(prompt, wait_s, accept_motion):
    robot_speak(prompt)
    wait_quiet()
    st.peak_motion = 0.0   # start measuring only after the robot finished speaking
    st.motion_frames = 0
    phrase = listen_phrase(wait_s, allow_inject=True)
    peak = round(st.peak_motion, 4)
    moved = st.motion_frames
    if is_distress(phrase):
        outcome = "DISTRESS"
    elif is_ok(phrase):
        outcome = "OK_VOICE"
    elif accept_motion and moved >= MOVEMENT_FRAMES:
        outcome = "OK_MOTION"
    else:
        outcome = "NO_RESPONSE"
    trace("checkin_result", outcome=outcome, heard=phrase, peak_motion=peak, moving_frames=moved)
    return {"outcome": outcome, "heard": phrase, "peak_motion": peak, "moving_frames": moved}


def encode_snapshot():
    frame = st.latest_frame
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def escalate(kind, evidence, events):
    st.robot_state = "EMERGENCY"
    if kind == "VOICE_DISTRESS":
        robot_speak("I am sorry you are not feeling well. Let me alert your doctor right away.")
    else:
        robot_speak("I could not confirm that you are safe. I am alerting your doctor now. Please stay calm.")

    ctx = {"kind": kind, "evidence": evidence, "age": patient["age"], "gender": patient["gender"],
           "condition": patient["condition"], "language": patient["language"], "events": events,
           "no_response": any(e.get("outcome") == "NO_RESPONSE" for e in events)}
    risk, summary = cloud.summarize(ctx)
    incident = cloud.create_incident(ctx, encode_snapshot(), risk, summary)
    cloud.send_email(patient["receiver"], patient["sender"], incident, ctx, risk, summary)
    trace("decision", action="escalate_to_physician", incident=incident["incident_id"],
          risk=risk, summary=summary)

    if not cloud.can_poll:
        trace("decision", action="no_cloud_polling", note="no incident store configured")
        return

    # Human-in-the-loop: wait for the physician's decision (or the patient cancelling).
    deadline = time.time() + cfg["doctor_wait"]
    reminded = False
    while time.time() < deadline:
        status = cloud.get_status(incident["incident_id"])
        if status == "DISPATCHED":
            trace("physician_decision", status=status)
            robot_speak("Your doctor has dispatched a medical team. Help is on the way. Please stay where you are.")
            st.robot_state = "EMERGENCY"
            return
        if status == "FALSE_ALARM":
            trace("physician_decision", status=status)
            robot_speak("Your doctor marked this as a false alarm. Resuming normal monitoring.")
            st.robot_state = "NORMAL_COMPANION"
            return
        phrase = listen_phrase(4, allow_inject=True)
        if phrase and is_ok(phrase):
            cloud.set_status(incident["incident_id"], "FALSE_ALARM", "patient")
            trace("patient_cancelled", heard=phrase)
            robot_speak("Thank you. I told your doctor that you are fine.")
            st.robot_state = "NORMAL_COMPANION"
            return
        if not reminded and time.time() > deadline - cfg["doctor_wait"] / 2:
            reminded = True
            cloud.send_email(patient["receiver"], patient["sender"], incident, ctx, risk, summary, reminder=True)
            trace("decision", action="reminder_email_sent")
            robot_speak("Your doctor has not answered yet. I have sent a reminder.")
    trace("decision", action="no_physician_ack")
    robot_speak("I have not heard from your doctor yet. Please call your emergency contact.")


def run_incident(kind, evidence, simulated=False):
    prof = PROFILE[kind]
    accept_motion = prof["accept_motion"] and not simulated  # hotkey tests: voice only
    events = []
    try:
        st.robot_state = "CHECKING_IN"
        trace("trigger", kind=kind, evidence=evidence)
        if kind != "VOICE_DISTRESS":
            res = check_in(prof["msg"], prof["wait"], accept_motion)
            events.append(res)
            if res["outcome"] == "NO_RESPONSE":
                res = check_in("I cannot hear you. If you are okay, say I am fine. "
                               "Otherwise I will alert your doctor.", 8, accept_motion)
                events.append(res)
            if res["outcome"] in ("OK_VOICE", "OK_MOTION"):
                trace("decision", action="cancel_alert", reason=res["outcome"])
                robot_speak("Thank you, I am glad you are okay.")
                st.robot_state = "NORMAL_COMPANION"
                return
        escalate(kind, evidence, events)
    finally:
        st.incident_active = False
        if pipeline:
            pipeline.rearm()


def start_incident(kind, evidence, simulated=False):
    if st.incident_active:
        return
    st.incident_active = True
    injected["phrase"] = ""
    threading.Thread(target=run_incident, args=(kind, evidence, simulated), daemon=True).start()


# ----------------------------------------------------------------------
# Privacy breaks (pause / sleep)
# ----------------------------------------------------------------------
def start_pause(mode):
    st.paused = True
    st.pause_mode = mode
    st.pause_until = time.time() + (cfg["sleep_s"] if mode == "sleep" else cfg["break_s"])
    st.robot_state = "SLEEPING" if mode == "sleep" else "PAUSED"
    if pipeline:
        pipeline.rearm()
    trace("pause_started", mode=mode, auto_resume_in_s=int(st.pause_until - time.time()))


def resume_monitoring(reason, speak=True):
    st.paused = False
    st.pause_mode = None
    st.robot_state = "NORMAL_COMPANION"
    if pipeline:
        pipeline.rearm()
    trace("pause_ended", reason=reason)
    if speak:
        robot_speak("Welcome back! Resuming your health monitoring." if reason == "patient"
                    else "Resuming your health monitoring now.")


# ----------------------------------------------------------------------
# Companion (voice commands, routines)
# ----------------------------------------------------------------------
def trigger_songs():
    robot_speak(random.choice([
        "Are you feeling nostalgic? Shall I play some golden old songs from Ghantasala or Balu sir?",
        "Hello! To make your time pleasant, shall I play some soothing old classic melodies?"]))


def trigger_quiz():
    question, answers = random.choice(QUIZ_QUESTIONS)
    quiz_state["answer"] = answers
    robot_speak("Let's refresh your mind with a quick quiz! " + question)


def trigger_news():
    robot_speak("Let's get updated with the world. Would you like to hear today's top news?")


def routine_prompts(hour):
    if st.paused:
        return
    if 6 <= hour < 9 and not prompt_tracker["morning_walk"]:
        prompt_tracker["morning_walk"] = True
        robot_speak("Good morning! I am online and watching over your walking safety.")
    elif 9 <= hour < 11 and not prompt_tracker["breakfast"]:
        prompt_tracker["breakfast"] = True
        robot_speak(f"It is breakfast time. Please take your meals and your pills for {patient['condition']}.")
    elif 12 <= hour < 16 and not prompt_tracker["songs"]:
        prompt_tracker["songs"] = True
        trigger_songs()


def handle_voice_command():
    global last_voice_time
    if st.listening or st.is_talking or st.incident_active:
        return
    if time.time() - last_voice_time < 5.0:
        return

    def work():
        global last_voice_time
        st.listening = True
        try:
            phrase = listen_phrase(4)
            last_voice_time = time.time()
            if not phrase:
                return
            st.last_voice = phrase
            print("Heard Voice:", phrase)

            # 1) SAFETY FIRST: distress always wins, even during a pause.
            if is_distress(phrase):
                if st.paused:
                    resume_monitoring("distress", speak=False)
                start_incident("VOICE_DISTRESS", f"patient said: {phrase}")
                return

            # 2) While paused only listen for "I am back".
            if st.paused:
                if has_any(phrase, RESUME):
                    resume_monitoring("patient")
                return

            # 3) Privacy breaks.
            if has_any(phrase, SLEEP):
                robot_speak("Entering sleep mode. I will stay quiet but still watch for falls. "
                            "Say, I am back, when you wake up.")
                start_pause("sleep")
                return
            if has_any(phrase, PAUSE):
                robot_speak("Taking a short privacy break. Say, I am back, when you return. "
                            "I will also resume by myself in thirty minutes.")
                start_pause("break")
                return

            # 4) Normal commands.
            if quiz_state["answer"]:
                answers = quiz_state["answer"]
                quiz_state["answer"] = None
                if has_any(phrase, answers):
                    robot_speak("Correct! Very good. Say quiz for another question.")
                else:
                    shown = next((a for a in sorted(answers) if a.isascii()), "unknown")
                    robot_speak(f"Good try! The answer was {shown}. Say quiz to try again.")
            elif has_any(phrase, QUIZ_WORDS):
                trigger_quiz()
            elif has_any(phrase, NEWS_WORDS):
                trigger_news()
            elif has_any(phrase, SONG_WORDS):
                trigger_songs()
            elif has_any(phrase, HELLO_WORDS):
                robot_speak("Hello there! I am your Chronos companion. How are you feeling right now?")
            elif has_any(phrase, OK):
                robot_speak("That is wonderful to hear! Remember to follow your daily routine.")
            else:
                robot_speak("I heard you. You can say pause, sleep, quiz, news, or songs.")
        finally:
            st.listening = False

    threading.Thread(target=work, daemon=True).start()


# ----------------------------------------------------------------------
# Onboarding + main loop
# ----------------------------------------------------------------------
def onboarding():
    print("\n==================================================")
    print("    CHRONOS CARE - AGENTIC VISION COMPANION")
    print("==================================================")
    print("AWS services:\n" + cloud.describe())

    patient["gender"] = input("\nEnter Patient Gender (Male/Female/Other): ").strip().capitalize() or "Unknown"
    while True:
        age = input("Enter Patient Age: ").strip()
        if age.isdigit() and 0 < int(age) < 120:
            patient["age"] = age
            break
        print("Invalid numeric age.")
    cond = input("\nHealth condition / history (e.g. Diabetes, Asthma, Knee Pain): ").strip()
    patient["condition"] = cond[:80] if cond else "General Wellness"

    patient["sender"] = input(f"\nVerified sender email [{patient['sender']}]: ").strip() or patient["sender"]
    patient["receiver"] = input(f"Doctor email [{patient['receiver']}]: ").strip() or patient["receiver"]

    print("\nHotkeys: s=Songs g=Quiz n=News | e=Distress f=Simulate fall i=Simulate immobile | y=say \"I am fine\" h=say \"help\" | r=Reset q=Quit")
    input("Press [ENTER] to start...")


def open_camera():
    for index in (0, 1, -1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
    return None


def main():
    global cloud, pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="short timers for the demo video")
    parser.add_argument("--skip-onboarding", action="store_true")
    parser.add_argument("--video", help="run on a video file instead of the webcam (loops)")
    parser.add_argument("--immobile", type=float, help="seconds of stillness before the first check-in")
    parser.add_argument("--facelost", type=float, help="seconds without a face before a check-in")
    args = parser.parse_args()

    cloud = CloudClient()
    if args.skip_onboarding:
        print("AWS services:\n" + cloud.describe())
    else:
        onboarding()

    detector = build_face_detector()
    if detector is None:
        print("No face model found. Run: python download_models.py   (folder:", BASE_DIR, ")")
        return

    immobile_s, face_lost_s = (8.0, 6.0) if args.demo else (30.0, 15.0)
    if args.immobile:
        immobile_s = args.immobile
    if args.facelost:
        face_lost_s = args.facelost
    cfg["doctor_wait"] = 90 if args.demo else 180
    if args.demo:
        cfg["break_s"], cfg["sleep_s"] = 60, 120   # short auto-resume for demos
    pipeline = PerceptionPipeline(detector, immobile_s=immobile_s, face_lost_s=face_lost_s)

    cap = cv2.VideoCapture(args.video) if args.video else open_camera()
    if cap is None or not cap.isOpened():
        print("Video file could not be opened!" if args.video else "Camera initialization failed! "
              "No webcam? Run with --video <file.mp4>")
        return
    fps = (cap.get(cv2.CAP_PROP_FPS) or 20.0) if args.video else 20.0
    frame_idx = 0
    print(f"OpenCV {cv2.__version__} | immobile>{immobile_s}s | face-lost>{face_lost_s}s | language={patient['language']}")
    robot_speak("Chronos companion is active. I am here with you.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if args.video and frame_idx > 0:      # loop the video so it can be watched again
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                pipeline.rearm()
                continue
            break
        if not args.video:
            frame = cv2.flip(frame, 1)            # mirror the webcam (clips are already mirrored)
        st.latest_frame = frame

        # Auto-resume so monitoring can never stay off forever.
        if st.paused and time.time() >= st.pause_until:
            resume_monitoring("timeout")

        # Break: ignore all triggers. Sleep: ignore stillness/face-lost but still watch for falls.
        if st.paused:
            pipeline.suppress = ({"IMMOBILE", "FACE_LOST"} if st.pause_mode == "sleep"
                                 else {"IMMOBILE", "FACE_LOST", "FALL_SUSPECTED"})
        else:
            pipeline.suppress = set()

        pr = pipeline.process(frame, (frame_idx / fps) if args.video else None)
        frame_idx += 1

        if st.incident_active:
            st.peak_motion = max(st.peak_motion, pr.motion)
            if pr.motion > MOVEMENT_RESPONSE:
                st.motion_frames += 1

        if pr.trigger and not st.incident_active:
            start_incident(pr.trigger, f"motion={pr.motion:.4f}, still={pr.immobile_s:.0f}s, "
                                       f"face_lost={pr.face_lost_s:.0f}s, fall_flag={pr.fall_suspected}")

        if not st.incident_active and st.robot_state != "EMERGENCY":
            if st.paused:
                st.robot_state = "SLEEPING" if st.pause_mode == "sleep" else "PAUSED"
            else:
                st.robot_state = "NORMAL_COMPANION" if pr.faces else "IDLE"
                if pr.faces:
                    routine_prompts(datetime.now().hour)

        # Listen when a face is visible, or when paused (to hear "I am back" / "help").
        if pr.faces or st.paused:
            handle_voice_command()

        for (x, y, w, h) in pr.faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if st.paused:
            left = max(0, int(st.pause_until - time.time()))
            status_text = f"{st.pause_mode.upper()} - auto-resume in {left}s"
            color = (0, 165, 255)
        else:
            status_text = f"motion={pr.motion:.3f} still={pr.immobile_s:.0f}s lost={pr.face_lost_s:.0f}s"
            color = (0, 255, 255)
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        ui = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_full_robot(ui, st.robot_state, time.time(), st.is_talking, st.message, patient["condition"])
        cv2.imshow("Chronos Care - Robot", ui)
        cv2.imshow("OpenCV Perception Feed", frame)

        key = cv2.waitKey(max(1, int(1000 / fps)) if args.video else 20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            trigger_songs()
        elif key == ord("g"):
            trigger_quiz()
        elif key == ord("n"):
            trigger_news()
        elif key == ord("e"):
            if st.paused:
                resume_monitoring("hotkey", speak=False)
            start_incident("VOICE_DISTRESS", "manual trigger (hotkey e)")
        elif key == ord("f"):
            start_incident("FALL_SUSPECTED", "simulated fall (hotkey f)", simulated=True)
        elif key == ord("i"):
            start_incident("IMMOBILE", "simulated immobility (hotkey i)", simulated=True)
        elif key == ord("y"):
            injected["phrase"] = "i am fine"
        elif key == ord("h"):
            injected["phrase"] = "help"
        elif key == ord("r") and not st.incident_active:
            if st.paused:
                resume_monitoring("hotkey", speak=False)
            st.robot_state = "IDLE"
            pipeline.rearm()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
