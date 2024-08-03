# MediWatch DVC pipeline

## For AWS ec2 Ubuntu
### 1. Install venv
```bash
sudo apt install python3-venv
```
### 2. Clone the repo
```
git clone git@github.com:shintotm/ik_capstone_v2.git
cd ik_capstone_v2/
```
### 3. Create and activate the virtual environment
```
python3 -m venv medi-env
echo "export PYTHONPATH=$PWD" >> medi-env/bin/activate
source medi-env/bin/activate
```
### 4. Install the python libraries
```
 pip install -r requirements.txt
```

### 5. Run the DVC Pipeline
```bash
dvc repro
```



