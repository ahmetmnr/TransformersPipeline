# -*- coding: utf-8 -*-
"""
Transformers Kütüphanesi - Pipeline Kullanımı Örnekleri
"""

# Gerekli kütüphaneyi içe aktaralım
from transformers import pipeline
import torch

def temel_pipeline_ornekleri():
    # GPU kullanımı için device değişkeni
    device = 0 if torch.cuda.is_available() else -1
    
    # 1. İngilizce Duygu Analizi
    duygu_analizi = pipeline("sentiment-analysis", device=device)
    ingilizce_ornek = "I've been waiting for a HuggingFace course my whole life."
    ingilizce_sonuc = duygu_analizi(ingilizce_ornek)
    
    # 2. Çok Dilli Duygu Analizi
    cokdilli_duygu = pipeline("sentiment-analysis", 
                               model="nlptown/bert-base-multilingual-uncased-sentiment",
                               device=device)
    turkce_ornek = "HuggingFace kursunu çok uzun zamandır bekliyordum."
    turkce_sonuc = cokdilli_duygu(turkce_ornek)
    
    # 3. Metin Oluşturma
    metin_olusturucu = pipeline("text-generation", device=device)
    baslangic_metni = "Artificial intelligence can"
    olusturulan_metin = metin_olusturucu(baslangic_metni, max_length=30, num_return_sequences=2)
    
    # 4. Maskeleme
    maskeleme = pipeline("fill-mask", device=device)
    maskeli_metin = "HuggingFace is creating a <mask> that the community uses to solve NLP tasks."
    tamamlanan_metinler = maskeleme(maskeli_metin)
    
    # Sonuçların gösterimi - bunları derste açıklayabilirsiniz
    return {
        "duygu_analizi_ingilizce": ingilizce_sonuc,
        "duygu_analizi_turkce": turkce_sonuc,
        "metin_olusturma": olusturulan_metin,
        "maskeleme": tamamlanan_metinler
    }

def gelismis_pipeline_ornekleri():
    # GPU kullanımı için device değişkeni
    device = 0 if torch.cuda.is_available() else -1
    
    # 5. Soru Cevaplama
    soru_cevaplama = pipeline("question-answering", device=device)
    soru = "What is HuggingFace?"
    metin = "HuggingFace is an AI company that develops tools for building applications using machine learning."
    soru_sonuc = soru_cevaplama(question=soru, context=metin)
    
    # 6. Metin Özetleme
    ozetleme = pipeline("summarization", device=device)
    uzun_metin = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals.
    """
    ozet = ozetleme(uzun_metin, max_length=75, min_length=30, do_sample=False)
    
    # 7. Çeviri
    ceviri = pipeline("translation_en_to_fr", device=device)
    ingilizce_metin = "HuggingFace is a company that provides NLP tools."
    fransizca_ceviri = ceviri(ingilizce_metin)
    
    # Sonuçların gösterimi - bunları derste açıklayabilirsiniz
    return {
        "soru_cevaplama": soru_sonuc,
        "ozetleme": ozet,
        "ceviri": fransizca_ceviri
    }

def turkce_dil_modelleri_ornekleri():
    # GPU kullanımı için device değişkeni
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        # 8. Türkçe BERT ile Maskeleme
        turkce_maskeleme = pipeline("fill-mask", model="dbmdz/bert-base-turkish-cased", device=device)
        turkce_maske = "Yapay zeka [MASK] alanında devrim yaratıyor."
        turkce_maske_sonuc = turkce_maskeleme(turkce_maske)
        
        # 9. Türkçe GPT ile Metin Üretimi
        # Not: Bu model yüklü değilse hata verebilir
        turkce_metin_uretimi = pipeline("text-generation", model="dbmdz/turkish-gpt2", device=device)
        turkce_baslangic = "Yapay zeka teknolojileri"
        turkce_uretilen = turkce_metin_uretimi(turkce_baslangic, max_length=40, num_return_sequences=1)
        
        return {
            "turkce_maskeleme": turkce_maske_sonuc,
            "turkce_metin_uretimi": turkce_uretilen
        }
    except Exception as e:
        return {"hata": f"Türkçe modeller yüklenirken bir hata oluştu: {e}"}

def siniflandirma_ornekleri():
    # GPU kullanımı için device değişkeni
    device = 0 if torch.cuda.is_available() else -1
    
    # 10. Metin Sınıflandırma
    siniflandirici = pipeline("text-classification", device=device)
    metinler = [
        "This movie is fantastic!",
        "I really hated the plot.",
        "The film was just okay, nothing special."
    ]
    siniflandirma_sonuclari = siniflandirici(metinler)
    
    # 11. Named Entity Recognition (Varlık İsmi Tanıma)
    ner = pipeline("ner", grouped_entities=True, device=device)
    metin = "My name is John and I work at Google in London."
    ner_sonucu = ner(metin)
    
    return {
        "siniflandirma": siniflandirma_sonuclari,
        "ner": ner_sonucu
    }

def pipeline_ozellikler():
    # GPU kullanımı için device değişkeni
    device = 0 if torch.cuda.is_available() else -1
    
    # 12. Pipeline Özelliklerini Gösterme
    # Bir modelin parametrelerini ayarlama
    duygu_parametreli = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased",
        framework="pt",  # PyTorch kullanımı
        device=device  # GPU kullanımı
    )
    sonuc = duygu_parametreli("I love this example!")
    
    # 13. Toplu İşlem (Batching)
    duygu_analizi_toplu = pipeline("sentiment-analysis", device=device)
    metinler = [
        "I love this product!",
        "This is not what I expected.",
        "The quality is amazing for the price."
    ]
    toplu_sonuclar = duygu_analizi_toplu(metinler)
    
    return {
        "parametreli_pipeline": sonuc,
        "toplu_islem": toplu_sonuclar
    }

def gpu_bilgisi_goster():
    """GPU bilgilerini göster"""
    if torch.cuda.is_available():
        print("\n===== GPU BİLGİSİ =====")
        print(f"Kullanılabilir GPU sayısı: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Aktif GPU: {torch.cuda.current_device()}")
        print(f"CUDA versiyonu: {torch.version.cuda}")
    else:
        print("\n❌ GPU bulunamadı. İşlemler CPU üzerinde gerçekleştirilecek.")

if __name__ == "__main__":
    # GPU bilgilerini göster
    gpu_bilgisi_goster()
    
    # Bu kısım sadece dosya doğrudan çalıştırıldığında çalışır
    # Temel örnekleri göster
    temel_sonuclar = temel_pipeline_ornekleri()
    for baslik, sonuc in temel_sonuclar.items():
        print(f"\n===== {baslik} =====")
        print(sonuc)
    
    # Gelişmiş örnekleri göster
    gelismis_sonuclar = gelismis_pipeline_ornekleri()
    for baslik, sonuc in gelismis_sonuclar.items():
        print(f"\n===== {baslik} =====")
        print(sonuc)
    
    # Türkçe dil modelleri örnekleri
    turkce_sonuclar = turkce_dil_modelleri_ornekleri()
    for baslik, sonuc in turkce_sonuclar.items():
        print(f"\n===== {baslik} =====")
        print(sonuc)
    
    # Sınıflandırma örnekleri
    siniflandirma_sonuclar = siniflandirma_ornekleri()
    for baslik, sonuc in siniflandirma_sonuclar.items():
        print(f"\n===== {baslik} =====")
        print(sonuc)
    
    # Pipeline özellikleri
    ozellik_sonuclar = pipeline_ozellikler()
    for baslik, sonuc in ozellik_sonuclar.items():
        print(f"\n===== {baslik} =====")
        print(sonuc) 