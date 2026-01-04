# Setup

## Create a virtual environment (recommended)
```py
python -m venv venv
source venv/bin/activate  # Linux/Mac


# Install dependencies
pip install -r requirements.txt
```

**2. Prepare your data**

Your folder structure should look like:
```
data/
├── aircraft/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── bird/
│   └── ...
├── drone/
│   └── ...
└── unknown/
    └── ...
```

# Getting Started:
1) Open project folder
2) For model training, enter this into the terminal: python train_cnn_autosplit.py --data_root data --pretrained --train_unknown 0 --num_workers 4
3) Test model output through the included jupyter notebook "test_model_notebook.ipynb"

Data Sources:  
https://www.kaggle.com/datasets/imbikramsaha/drone-bird-classification  
https://www.kaggle.com/datasets/dmcgow/birds-200  
https://www.kaggle.com/datasets/cybersimar08/drone-detection  
https://www.kaggle.com/datasets/sonainjamil/malicious-drones  
https://www.kaggle.com/datasets/dolphinramses/clouds  
https://www.kaggle.com/datasets/serhiibiruk/balloon-object-detection  


# Gradio?
```py
# Just run in vsc
python app.py
```