<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Webcam Document Detector Project

This is a React TypeScript project with Vite for real-time document and face detection using ONNX.js and webcam.

## Project Context

-   Uses ONNX.js for running YOLOX model inference in the browser
-   Implements real-time webcam feed processing
-   Detects 'face' and 'doc_quad' classes
-   Displays bounding boxes and confidence scores
-   Shows detailed detection metrics in a sidebar

## Key Technologies

-   React 18 with TypeScript
-   Vite for fast development
-   ONNX.js for model inference
-   Canvas API for drawing bounding boxes
-   WebRTC for webcam access

## Code Style Preferences

-   Use functional components with hooks
-   Prefer TypeScript interfaces for type safety
-   Use modern ES6+ syntax
-   Implement proper error handling for webcam and model loading
-   Keep components modular and reusable
