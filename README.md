# DirectionNet Linux Kurulum Kılavuzu

Bu doküman, sıfırdan kurulmuş (Python dahil olmayan) bir **Linux** sistemde DirectionNet projesini çalıştırmak için gerekli kurulum adımlarını içerir.

## 1) Sistem Gereksinimleri

Aşağıdaki araçlar gereklidir:

- `git`
- `curl`
- `unzip`
- Temel derleme araçları (`build-essential` gibi)

Debian/Ubuntu tabanlı sistemlerde:

```bash
sudo apt update
sudo apt install -y git curl unzip build-essential
```

## 2) Python Kurulumu (Miniconda)

Sistemde Python olmadığı için Miniconda ile izole bir Python ortamı kuracağız.

```bash
mkdir -p "$HOME/miniconda3"
curl -L -o Miniconda3-latest-Linux-x86_64.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash
```

> ARM tabanlı Linux kullanıyorsanız `Miniconda3-latest-Linux-aarch64.sh` paketini tercih edin.

## 3) Depoyu Klonlama

```bash
git clone git@github.com:cahitbarankilinc/DirectionNet_Setup_For_MacOS.git
cd DirectionNet_Setup_For_MacOS
```

## 4) Python Ortamı Oluşturma

```bash
conda create -n directionnet_baran python=3.11 -y
conda activate directionnet_baran
```

## 5) Gerekli Python Paketleri

TensorFlow ve bağımlılıklarını kurun:

```bash
python -m pip install -U pip
python -m pip install "tensorflow==2.15.0"
python -m pip install "keras==2.15.0"
python -m pip install "tensorflow-probability==0.23.0"
python -m pip install "tf-slim==1.1.0" "tensorflow-graphics"
```

> GPU ile çalışmak isterseniz NVIDIA CUDA/cuDNN uyumlu sürümleri kurmanız gerekir. Bu repo CPU ile de çalışır.

## 6) Kurulum Kontrolü

```bash
python -c "import tensorflow as tf, sys; print('TF:', tf.__version__); print('PY:', sys.executable)"
```

Beklenen çıktı örneği:

```bash
TF: 2.15.0
PY: /home/<kullanici>/miniconda3/envs/directionnet_baran/bin/python
```
