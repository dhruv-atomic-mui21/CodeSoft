# AI Recommendation System - Backend

## Project Overview
This is the backend service for an AI-powered recommendation system. It uses the Gemini API (Google's generative language model) to generate personalized recommendations based on user-selected categories and preferences.

## Features
- Flask-based REST API backend.
- Integrates with Gemini API to generate top 5 recommendations.
- Handles retries and error responses gracefully.
- Serves a frontend interface for user interaction.

## Setup Instructions
1. Ensure Python and pip are installed.
2. Install dependencies (Flask, requests, python-dotenv).
3. Set the environment variable `API_KEY` with your Gemini API key in a `.env` file or system environment.
4. Run the backend server:
   ```
   python app.py
   ```

## Usage
- Access the frontend by navigating to `http://localhost:5000/` in your browser.
- Select a category and enter preferences to get AI-generated recommendations.

## API Details
- POST `/recommend`: Accepts JSON with `category` and `preferences` fields, returns recommendations in JSON format.
