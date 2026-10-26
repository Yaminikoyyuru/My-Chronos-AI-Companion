# ChronosAI Companion — Technical Report

## 1. Overview
ChronosAI Companion is an elder-care monitoring application that uses OpenCV 5 and AWS to identify potential safety incidents and support timely review.

## 2. Problem Statement
Falls or prolonged immobility may go unnoticed when older adults are alone. This project aims to support monitoring through visual event detection, voice check-ins, and incident alerts.

## 3. Implementation
OpenCV 5 processes camera or video frames to detect configured events, including suspected falls, immobility, and face loss. The application can initiate voice check-ins and escalate incidents when safety cannot be confirmed.

## 4. AWS Integration
- **Amazon S3:** Stores incident records, snapshots, review pages, and decisions.
- **Amazon SES:** Sends incident email notifications.
- **Amazon CloudWatch:** Supports logging when configured and available.
- **Amazon Bedrock:** Optional risk-summary generation, subject to permissions.

## 5. Workflow
Camera/video → OpenCV 5 → Event detection → Voice check-in → Incident creation → AWS storage and email alert → Doctor decision → Application response.

## 6. Testing and Limitations
The application has undergone repeated development testing. The demonstration shows incident notification and doctor decision handling. Detection errors, camera conditions, background noise, network issues, and AWS availability may affect performance. A recorded dispatch decision does not guarantee that emergency services physically arrive.

## 7. Privacy and Conclusion
Protect AWS credentials and sensitive images, restrict access to incident records, and apply suitable data-retention controls. ChronosAI Companion is an assistive prototype, not a replacement for professional medical care or verified emergency services.

## Links
- Repository: https://github.com/Yaminikoyyuru/My-Chronos-AI-Companion
- Demo: https://vimeo.com/1239531061?share=copy&fl=sv&fe=ci
