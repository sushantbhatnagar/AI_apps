# Test Cases Generator

This uses python streamlit library as web-ui interface and lets the user upload the requirement document. 
Based on the requirement document uploaded, user is able to query the llm model to generate test cases for the system under test. 


## Getting started

### Before you begin you will need:
- A decent Python IDE like PyCharm.
- Python 3.10 or higher.


### Clone the repo

```shell
git clone  git@github.com:sushantbhatnagar/AI_apps.git
```


#### Create an Python virtual environment in the project root.

```shell
python -m venv venv
```

After this there will be a `venv` sub-folder containing a clean Python installation 

While still in that same command line you will want to activate the environment.  

- On Linux/MacOS/WSL run: `source venv/bin/activate`
- If you're on Windows run: `venv\Scripts\activate.bat`

Still in that same command line run the following to pull down all the dependencies


```shell
pip install -r requirements.txt 
```

Make sure to set the python interpreter in your IDE to the one now in `<project_root>/venv`. 


## Get an Open AI API Key.
Visit the signup page for the [Open AI Platform](https://platform.openai.com/signup) to sign up. Once you have the API_KEY, create a .env file and enter: 
OPENAI_API_KEY=<your-openai-key>


# Run the app
In the terminal, run the following command
```
streamlit run test_generator.py
```

