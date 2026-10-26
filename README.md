
# ChronosAI Companion

**An OpenCV 5 and AWS-based elder-care monitoring prototype.**

## Overview
ChronosAI Companion monitors configured visual events and supports safety check-ins, incident recording, email notifications, and doctor decision handling.

## Key Features
- Camera/video monitoring using OpenCV 5
- Configured event detection, including suspected falls and immobility
- Voice-based safety check-ins
- Incident records and snapshots stored in Amazon S3
- Email alerts using Amazon SES
- Doctor review and decision recording
- Optional AI risk summaries, subject to AWS permissions and configuration

## Technology Stack
- Python
- OpenCV 5
- Amazon S3
- Amazon SES
- Amazon CloudWatch, when configured

## Workflow
Camera/video → Visual event detection → Voice check-in → Incident creation → AWS storage and email alert → Doctor decision → Application response.

## Repository Documentation
- `docs/architecture.png` — System architecture diagram
- `docs/technical_report.md` — Technical report
- `docs/evaluation.md` — Evaluation report
- `docs/evaluation/` — Evidence screenshots

*Adjust documentation paths to match the actual filenames in this repository.*

## Demo
https://vimeo.com/1239531061?share=copy&fl=sv&fe=ci

## Setup
1. Install a compatible Python version.
2. Install the dependencies listed in `requirements.txt`.
3. Configure AWS credentials securely outside the repository.
4. Configure the required AWS permissions and application settings.
5. Follow the project's actual entry-point and model setup instructions to run the application.

## Limitations and Responsible Use
This is an assistive prototype, not a medical diagnostic system or a verified emergency-response service. Visual detection may produce false alarms or miss events. Protect camera images, incident records, and AWS credentials.

## Security
Never commit `.env` files, AWS secret keys, passwords, or private patient information.

