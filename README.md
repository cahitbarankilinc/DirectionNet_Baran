# DirectionNet Windows Kurulum Kılavuzu

Bu kılavuz, Python kurulu olmayan sıfır Windows makinede DirectionNet'i çalıştırmak için gereken tüm adımları içerir. Yalnızca Windows içindir.

## 1) Gerekli Araçlar

### 1.1 Git Kurulumu
1. [Git for Windows](https://git-scm.com/download/win) indirin ve kurun.
2. Kurulum sırasında varsayılan seçenekleri kullanabilirsiniz.

### 1.2 Miniforge (Conda) Kurulumu
1. [Miniforge3 Windows (x86_64)](https://github.com/conda-forge/miniforge/releases/latest) dosyasını indirin. Dosya adı genelde `Miniforge3-Windows-x86_64.exe` şeklindedir.
2. Kurulumda **"Add Miniforge3 to PATH"** seçeneğini işaretleyin.
3. Kurulum bittikten sonra **PowerShell** açın.

### 1.3 Kurulum Doğrulama
PowerShell'de aşağıdaki komutları çalıştırın:
```powershell
conda --version
```
Çıktıda conda sürümü görünmelidir.

## 2) Projeyi Klonlama
PowerShell'de aşağıdaki komutları çalıştırın:
```powershell
git clone https://github.com/cahitbarankilinc/DirectionNet_Setup_For_MacOS.git DirectionNet_Baran
cd DirectionNet_Baran
```

> Not: Depo adı MacOS olarak görünse de kod Windows'ta da çalışacak şekilde kurulabilir.

## 3) Python Ortamını Oluşturma
```powershell
conda create -n directionnet_baran python=3.11 -y
conda activate directionnet_baran
```

## 4) Gerekli Kütüphaneleri Kurma
Önce pip'i güncelleyin:
```powershell
python -m pip install --upgrade pip
```

Ardından TensorFlow ve bağımlılıkları kurun:
```powershell
python -m pip install "tensorflow==2.15.0"
python -m pip install "keras==2.15.0"
python -m pip install "tensorflow-probability==0.23.0"
python -m pip install "tf-slim==1.1.0" "tensorflow-graphics"
```

Kurulumu test etmek için:
```powershell
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

> Windows üzerinde GPU desteği farklılık gösterebilir. GPU görünmüyorsa CPU ile çalışır.

## 5) Dataset İndirme
Aşağıdaki iki dataset'i indirip `data` klasörüne çıkarın:

- [**MatterportA test data**](https://drive.google.com/file/d/1be75Ys8vi1o7eeS_Rf0SuJxlTkDJNisZ/view?usp=sharing)
- [**MatterportB test data**](https://drive.google.com/file/d/1PcyD_8TZOOKh6G8B8eUHQrOUEOMrMx_F/view?usp=sharing)

Klasör yapısı aşağıdaki gibi olmalıdır:
```
DirectionNet_Baran/
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

## 6) Model Eğitimi
DirectionNet-R eğitimi için örnek komut:
```powershell
python -u train.py `
  --checkpoint_dir checkpoints/R `
  --data_dir data/MatterportA/test `
  --model 9D `
  --batch 2
```

### 6.1 Directional Transformer Kullanımı ve Güçlendirme
- Directional transformer **yalnızca** `--model 9D` ve `--model Single` seçimlerinde aktiftir.
- `--model 6D` ve `--model T` eğitimlerinde transformer kullanılmaz (flag açık olsa bile).

Aşağıdaki bayraklarla transformer boyutunu büyütebilirsiniz:
```powershell
python -u train.py `
  --checkpoint_dir checkpoints/R `
  --data_dir data/MatterportA/test `
  --model 9D `
  --enable_directional_transformer True `
  --transformer_hidden_size 1024 `
  --transformer_num_heads 16 `
  --transformer_mlp_dim 4096 `
  --transformer_num_layers 8 `
  --transformer_dropout 0.1 `
  --batch 2
```

> Not: Bu ayarlar çok daha büyük GPU bellek ihtiyacı yaratır. `transformer_*` bayrakları ile daha küçük veya daha büyük modeller kurabilirsiniz.

> **Not:** Directional transformer, `tf.nn.gelu` veya `keras.activations.gelu` bulunmayan ortamlarda otomatik olarak yerel GELU yaklaşımını kullanır.

## 7) Değerlendirme
```powershell
python eval.py `
  --checkpoint_dir checkpoints/R `
  --eval_data_dir data/MatterportA `
  --save_summary_dir logs/eval_R `
  --testset_size 1000 `
  --batch 8 `
  --model 9D
```

## 8) Kaynak
- [**Original Source**](https://github.com/arthurchen0518/DirectionNet?tab=readme-ov-file)
