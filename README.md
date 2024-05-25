# How to Run

## Step 1: Download Python, Anaconda, and Postman
Download Python from this [link](https://www.python.org/downloads) 🐍

Download Anaconda from this [link](https://docs.anaconda.com/free/anaconda/install/mac-os/) 🌳

Download Postman from this [link](https://www.postman.com/downloads/) 👷

## Step 1.5 Add Tensorboard Extension to Visual Studio Code (Optional)
You can add the extension directly inside VS Code, or by using this [link](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.tensorboard) 📊

## Step 2: Create & Activate Conda Environment With Python 3.11 or lower
LightFM is the only component limiting Python version, if the model is switched out, keeping Python version and packages up to date is advisable.
```bash
conda create -n ml python=3.11
```
```bash
conda activate ml
```

## Step 3: Install Dependencies in the Environment
```bash
pip install -r requirements.txt
```

## Step 4: Host the API
Update IP to your local IP address and run model serving API
```bash
python lightfm_api.py
```

## Step 5: Manually test API Endpoints (or just run tests)
Go to Postman GUI and make a new request for `0.0.0.0:8080`
- GET Request: `/{endpoint}`
- POST Request: `/{endpoint}?{parameter_name}={data}`

# Definition of Done
## Functionality
- Code must be functional, no new errors should be intentionally introduced to master branch (test the dev branch after committing PR before making PR to master)
  
## Code
- A PBI is not done before it is on the dev branch without any new errors
- No hard-coded variables or dead code
- Function/Variable names should be sufficient to understand their purpose (if not, write a comment at declaration)
- Close a feature branch once the feature has been merged into dev
- Any new dependencies should be clearly documented
- (Actions Check): Linting

## Testing
- (Actions Check): All tests must pass
- Test code when it bears significant functionality (no frontend testing)
- All code must be reviewed by at least one other team member

## Design
- All frontend components and their assets / utilities must be located in the correct folders
- If a design guide is provided, the design must follow said guide
- All design must be tested on Expo Go (or emulator)
  
## Best practices
- All routes must have exception handling
- All PRs should clearly indicate what PBI they belong to

# Short version for Github (use in PR Template)
- [x] Code has been tested locally and passes all relevant tests.
- [x] Documentation has been updated to reflect the changes, if applicable.
- [x] Code follows the established coding style and guidelines of the project.
- [x] All new and existing tests related to the changes have passed. 
- [x] Any necessary dependencies or new packages have been properly documented.
- [x] Pull request title and description are clear and descriptive.
- [x] Reviewers have been assigned to the pull request.
