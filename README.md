# Usage

## Dependencies

Please make sure you've installed the following in your execution environment using `pip install`
`llama_index`, `uvicorn`, `fastapi`

## How to run

Run `main.py` on localhost using port 8000

Open `main.html` which will connect to the backend and provide UI for essential functions.

You can interact with the backend using the UI in the browser when the HTML page is loaded.

The directory you cloned this repo to will be used to store documents and indexes generated.

Make sure you have a `.env` file in the directory with the environment variable `API_KEY` from your Hugging Face account.

## Switching between LLMs

You can use the drop down menu to switch between models.

Adding future models is straightforward: just add them in the selection menu in the HTML code.

Alternatively, a standalone version of this app can be produced where the AI models are simply strings of the names on HuggingFace.

Please bear in mind, that not all AI models are suited to this type of problem and some can throw errors or not work at all.


