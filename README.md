# 1)DirectionNet Installation Guide (For MacOS)

First, clone the repository to your local computer:

```bash
git clone git@github.com:cahitbarankilinc/DirectionNet_Setup_For_MacOS.git
```
The Python files in this repository are set up to work on MacOS operating systems.

<br><br>

# 2)Downloading the Datasets

After setting up the repository locally, download the following two datasets:

- [**MatterportA test data**](https://drive.google.com/file/d/1be75Ys8vi1o7eeS_Rf0SuJxlTkDJNisZ/view?usp=sharing)
- [**MatterportB test data**](https://drive.google.com/file/d/1PcyD_8TZOOKh6G8B8eUHQrOUEOMrMx_F/view?usp=sharing)

These datasets will be downloaded as `.zip` files. Extract the zip files and place them in the `/data` folder of the repository as shown below:

```bash
├ train.py
├ eval.py
├ dataset/
├ data/
│
├── MatterportA/
│   ├ README
│   ├ test/
│   ├ test_meta/
│
├── MatterportB/
│   ├ README
│   ├ test/
│   ├ test_meta/
```

<br><br>

# 3Installation Steps

## 1️⃣ Arm64 Miniforge Installation

Before setting up the repository, create a new folder (do **not** clone the repo yet). Then follow these steps in order:

### Download & Install Miniforge:
```bash
curl -L -o Miniforge3-MacOSX-arm64.sh \
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh -b -p "$HOME/miniforge_arm"
```
### For Conda init (arm):
```bash
"$HOME/miniforge_arm/bin/conda" init zsh
exec zsh
```


## 2️⃣ Verification

After completing the installation, close and reopen the terminal. Navigate to the directory you created for the repository and verify with the following commands:
```bash
which conda
conda info | egrep "platform|arch|base environment"
```
### Expected results:
```bash
platform: osx-arm64
arch: arm64
base environment: .../miniforge_arm
```


## 3️⃣ Creating Environment
```bash
conda create -n directionnet_baran python=3.11 -y
conda activate directionnet_baran
```



## 4️⃣ Installing TensorFlow and Libraries
First, remove any old TensorFlow versions:
```bash
python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-intel || true
python -m pip cache purge
python -m pip install -U pip
```

### Install the correct packages for Apple Silicon:
```bash
python -m pip install "tensorflow==2.15.0"
python -m pip install tensorflow-metal
python -m pip install "keras==2.15.0"
python -m pip install "tensorflow-probability==0.23.0"
python -m pip install "tf-slim==1.1.0" "tensorflow-graphics"
```

### Test it:
```bash
python -c "import tensorflow as tf, platform, sys; print('Arch:', platform.machine()); print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU')); print('PY:', sys.executable)"

```
### Expected results:
```bash
Arch: arm64
TF: 2.15.0 (or 2.1x)
GPU:  <Apple Metal in list>
PY:  /Users/<username>/miniforge_arm/
```



<br><br>



# 🚀 Model Training

You can now start training the model. Below are example commands to run the training:
### Train DirectionNet-R
```bash
python -u train.py \
  --checkpoint_dir checkpoints/R \
  --data_dir data/MatterportA/test \
  --model 9D \
  --batch 2
```

### Evaluation DirectionNet-R
```bash
python eval.py \
  --checkpoint_dir checkpoints/R \
  --eval_data_dir data/MatterportA \
  --save_summary_dir logs/eval_R \
  --testset_size 1000 \
  --batch 8 \
  --model 9D
```

## Transformer'lı ortak mod için kısacık özet
- **`9D+T` model seçeneği**: Üç rotasyon kafasına ek bir çeviri kafası üretir. Bu yapı sayesinde transformer, rotasyon ve çeviri token’larını birlikte iyileştirir.
- **Bayrak**: `--enable_directional_transformer=True` varsayılan olarak açık, kapatmak isterseniz `False` geçebilirsiniz.
- **Kullanım örneği (eğitim)**:
  ```bash
  python -u train.py \
    --checkpoint_dir checkpoints/R_joint \
    --data_dir data/MatterportA/test \
    --model 9D+T \
    --batch 2
  ```
- **Kullanım örneği (değerlendirme)**:
  ```bash
  python eval.py \
    --checkpoint_dir checkpoints/R_joint \
    --eval_data_dir data/MatterportA \
    --save_summary_dir logs/eval_R_joint \
    --testset_size 1000 \
    --batch 8 \
    --model 9D+T \
    --enable_directional_transformer=True
  ```
- **Gereken veri**: `dataset_loader` rotasyon yanında çeviri yer tutucu da döndürdüğünden ek veri hazırlığı gerekmiyor; sadece `--model 9D+T` seçimi rotasyon+çeviri kayıplarını birlikte optimize ediyor.



<br><br>

# For More: 
- [**Original Source**](https://github.com/arthurchen0518/DirectionNet?tab=readme-ov-file)
