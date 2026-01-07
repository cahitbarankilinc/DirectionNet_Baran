# DirectionNet Linux Kurulum Kılavuzu

Bu doküman, Python yüklü olmayan sıfırdan bir Linux kurulumunda DirectionNet'i çalıştırmak için gerekli adımları içerir. Yalnızca Linux için hazırlanmıştır.

## 1) Sistem Bağımlılıkları (Python yokken)

Aşağıdaki komutlar Debian/Ubuntu tabanlı dağıtımlar içindir. Farklı bir dağıtım kullanıyorsanız eşdeğer paketleri kurun.

```bash
sudo apt update
sudo apt install -y git curl ca-certificates build-essential
```

## 2) Miniforge (Conda) Kurulumu

Python yüklü olmadığı için Miniforge ile izole bir ortam oluşturacağız.

```bash
curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge"
```

Conda'yı etkinleştirmek için shell başlatma dosyanızı güncelleyin:

```bash
"$HOME/miniforge/bin/conda" init bash
exec bash
```

Kurulumu doğrulayın:

```bash
which conda
conda info | egrep "platform|arch|base environment"
```

Beklenen örnek çıktı:

```bash
platform: linux-64
arch: x86_64
base environment: .../miniforge
```

> **Not:** ARM tabanlı Linux (aarch64) kullanıyorsanız `Miniforge3-Linux-aarch64.sh` indirin.

## 3) Projeyi İndirme

```bash
git clone git@github.com:cahitbarankilinc/DirectionNet_Setup_For_MacOS.git
cd DirectionNet_Setup_For_MacOS
```

## 4) Conda Ortamı Oluşturma

```bash
conda create -n directionnet_baran python=3.11 -y
conda activate directionnet_baran
```

## 5) TensorFlow ve Bağımlılıkların Kurulumu

Önce olası eski paketleri temizleyin ve pip'i güncelleyin:

```bash
python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-intel || true
python -m pip cache purge
python -m pip install -U pip
```

Linux için önerilen paketler:

```bash
python -m pip install "tensorflow==2.15.0"
python -m pip install "keras==2.15.0"
python -m pip install "tensorflow-probability==0.23.0"
python -m pip install "tf-slim==1.1.0" "tensorflow-graphics"
```

Kurulumu test edin:

```bash
python -c "import tensorflow as tf, platform, sys; print('Arch:', platform.machine()); print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU')); print('PY:', sys.executable)"
```

## 6) Veri Setlerini İndirme

Aşağıdaki iki veri setini indirin:

- [**MatterportA test data**](https://drive.google.com/file/d/1be75Ys8vi1o7eeS_Rf0SuJxlTkDJNisZ/view?usp=sharing)
- [**MatterportB test data**](https://drive.google.com/file/d/1PcyD_8TZOOKh6G8B8eUHQrOUEOMrMx_F/view?usp=sharing)

`.zip` dosyalarını çıkarın ve aşağıdaki gibi `/data` klasörüne yerleştirin:

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

## 7) Eğitim ve Değerlendirme

### DirectionNet-R Eğitimi

```bash
python -u train.py \
  --checkpoint_dir checkpoints/R \
  --data_dir data/MatterportA/test \
  --model 9D \
  --batch 2
```

> **Not:** Directional transformer, `tf.nn.gelu` veya `keras.activations.gelu` bulunmayan TensorFlow kurulumlarında uyumluluk için yerel bir GELU yaklaşımı kullanır. Ek yapılandırma gerektirmez.

### DirectionNet-R Değerlendirme

```bash
python eval.py \
  --checkpoint_dir checkpoints/R \
  --eval_data_dir data/MatterportA \
  --save_summary_dir logs/eval_R \
  --testset_size 1000 \
  --batch 8 \
  --model 9D
```

## Daha Fazlası

- [**Orijinal Kaynak**](https://github.com/arthurchen0518/DirectionNet?tab=readme-ov-file)
