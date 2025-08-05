# COMPASS - GDPR Privacy Policy Auditor

This project is an AI-powered tool to analyze privacy policies for GDPR compliance.

## Prerequisites

- You must have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your machine.

## Quick Start

1.  **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd compass-app
    ```

3.  **Start the application:**
    ```sh
    docker-compose up --build
    ```
    The first time you run this, Docker will build the image, which may take several minutes to download all the libraries and models.

4.  **Test the API:**
    - Once the build is complete and the server is running, open your web browser and go to `http://127.0.0.1:8000/docs`.
    - You can now use the interactive API documentation to test the `/analyze/file` endpoint.