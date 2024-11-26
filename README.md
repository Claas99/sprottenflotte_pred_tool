# sprottenflotte_pred_tool
<<<<<<< HEAD

A prediction model for sprottenflotte/kielregion

Structure:

```
data.py
data_temp.csv
prediction.py
predictions.csv
station_data.csv
 
requirements.txt
README.md
.gitignore
sample.env
```

This lauches a website


Python scripts functionality:




## Step 1: Installation:

### 1. Clone the repository from GitHub

Clone the repository to a nice place on your machine via:

```
git clone git@github.com:Claas99/sprottenflotte_pred_tool.git
```

### 2. Create a new environment

#### 2.1 Virtual environment

Make a new virtual .venv environment with python version 3.12 or an conda environment.

#### 2.2 Conda environment

Create a new Conda environment for this project with a specific version of Python:

```
conda create --name application_project_2024 python=3.12
```

Initialize Conda for shell interaction:

To make Conda available in you current shell execute the following:

```
conda init
```

### 3. Install packages from requirements.txt

Install packages in environment:

```
pip install -r requirements.txt
```

## Step 2: Create a .env file

Create a .env file in the project directory.

This .env file will store sensitive information such as passwords, email and client secrets securely. The .env file is not under version control and therefore secure, because it is added to the .gitignore file.

**Note:** You can use the `sample.env` file, which serves as a template, for creating your own `.env` file.

```
# password for keycloak
PASSWORD="*************"

# client_secret
CLIENT_SECRET="fP81XZ5OTt5iRJ7qhyyTCv4eQtpGqc5i"

# email username for addix API
USERNAME_EMAIL='email@website.de'

```
=======
A prediction model for sprottenflotte/kielregion
>>>>>>> ffab171956a1d4ddfe6198014864e737cc81a0e3
