# BusPredict Manual Testing Guide

## 1. Purpose

This guide explains how to manually test the BusPredict prototype, including its backend API and browser dashboard.

## 2. Prerequisites

- Python installed.
- Required project dependencies installed.
- Project files and the required trained model available.

## 3. Start the Backend

1. Open a terminal in the project folder.
2. Follow the setup instructions in the README.
3. Start the FastAPI server using the documented command.
4. Check the terminal for errors.

## 4. Open the Dashboard

1. Open `frontend/index.html` in a browser.
2. Confirm that the dashboard loads.
3. Check the browser console if errors appear.

## 5. Manual Test Cases

| Test | Action | Expected result |
|---|---|---|
| Backend startup | Start the API | The server starts without errors |
| API documentation | Open `/docs` on the local API server | API documentation loads |
| Dashboard loading | Open the HTML dashboard | The interface appears |
| Valid prediction | Submit valid sample data | A prediction response is returned |
| Invalid input | Submit invalid or incomplete data | The application handles the request appropriately |

## 6. Test Results



