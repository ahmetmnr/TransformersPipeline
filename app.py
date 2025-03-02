# -*- coding: utf-8 -*-
"""
Transformers Kütüphanesi - Flask Tabanlı Pipeline Demo Arayüzü
"""

from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sentence_transformers import SentenceTransformer, util
import base64
import soundfile as sf
import os
import tempfile
from werkzeug.utils import secure_filename

# GPU kullanımı için device değişkeni
device = 0 if torch.cuda.is_available() else -1

# Flask uygulamasını oluştur
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Modelleri önbellekte tutan sınıf
class PipelineCache:
    def __init__(self):
        self.pipelines = {}
    
    def get_pipeline(self, task, model=None, **kwargs):
        key = f"{task}_{model}" if model else task
        if key not in self.pipelines:
            # Belleği temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            # Pipeline oluştur
            self.pipelines[key] = pipeline(task, model=model, device=device, **kwargs)
        return self.pipelines[key]

# Pipeline önbelleği oluştur
pipeline_cache = PipelineCache()

# Ana sayfa
@app.route('/')
def index():
    # GPU bilgisi
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
    else:
        gpu_info = "GPU bulunamadı. İşlemler CPU üzerinde çalışacak."
    
    return render_template('index.html', gpu_info=gpu_info)

# Duygu analizi API
@app.route('/api/duygu-analizi', methods=['POST'])
def duygu_analizi_api():
    data = request.json
    metin = data.get('metin', '')
    dil = data.get('dil', 'en')
    
    if not metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    try:
        # Model seçimi
        if dil == "en":
            model = "distilbert-base-uncased-finetuned-sst-2-english"
        else:
            model = "nlptown/bert-base-multilingual-uncased-sentiment"
        
        # Önbellekten pipeline al
        duygu_analizi = pipeline_cache.get_pipeline("sentiment-analysis", model=model)
        sonuc = duygu_analizi(metin)
        
        # Sonuç sınıfını belirle
        label = sonuc[0]["label"]
        score = sonuc[0]["score"]
        sentiment_class = "positive" if "POSITIVE" in label else "negative" if "NEGATIVE" in label else "neutral"
        
        return jsonify({
            "label": label,
            "score": float(score),
            "sentiment_class": sentiment_class,
            "metin": metin
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Metin üretme API
@app.route('/api/metin-uretme', methods=['POST'])
def metin_uretme_api():
    data = request.json
    baslangic_metni = data.get('baslangic_metni', '')
    max_uzunluk = int(data.get('max_uzunluk', 50))
    tekrar_sayisi = int(data.get('tekrar_sayisi', 1))
    
    if not baslangic_metni.strip():
        return jsonify({"error": "Lütfen bir başlangıç metni girin!"})
    
    try:
        # Önbellekten pipeline al
        metin_uretici = pipeline_cache.get_pipeline("text-generation")
        sonuclar = metin_uretici(
            baslangic_metni, 
            max_length=max_uzunluk,
            num_return_sequences=tekrar_sayisi,
            do_sample=True
        )
        
        return jsonify({
            "sonuclar": [sonuc['generated_text'] for sonuc in sonuclar]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Maskeleme API
@app.route('/api/maskeleme', methods=['POST'])
def maskeleme_api():
    data = request.json
    maskeli_metin = data.get('maskeli_metin', '')
    
    if not maskeli_metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    try:
        # Hangi mask token'ı kullanıldığını belirleyelim
        if "[MASK]" in maskeli_metin:
            mask_token = "[MASK]"
            model = "bert-base-uncased"
        elif "<mask>" in maskeli_metin:
            mask_token = "<mask>"
            model = "roberta-base"
        else:
            return jsonify({"error": "Lütfen maskelenmiş bir kelime ekleyin: [MASK] veya <mask>"})
        
        # Önbellekten pipeline al
        maskeleme = pipeline_cache.get_pipeline("fill-mask", model=model)
        sonuclar = maskeleme(maskeli_metin)
        
        return jsonify({
            "sonuclar": [{
                "sequence": sonuc['sequence'],
                "score": float(sonuc['score']),
                "token": sonuc['token_str']
            } for sonuc in sonuclar[:5]]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Soru cevaplama API
@app.route('/api/soru-cevaplama', methods=['POST'])
def soru_cevaplama_api():
    data = request.json
    soru = data.get('soru', '')
    metin = data.get('metin', '')
    
    if not soru.strip():
        return jsonify({"error": "Lütfen bir soru girin!"})
    
    if not metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    try:
        # Model yükle
        model_name = "deepset/roberta-base-squad2"
        
        # Önbellekten pipeline al
        soru_cevaplama = pipeline_cache.get_pipeline("question-answering", model=model_name)
        
        # Soruyu cevapla
        sonuc = soru_cevaplama(question=soru, context=metin)
        
        # Cevap yoksa veya güven düşükse
        if not sonuc["answer"] or sonuc["score"] < 0.01:
            cevap = "Metinde bu sorunun cevabı bulunamadı."
        else:
            cevap = sonuc["answer"]
        
        return jsonify({
            "soru": soru,
            "cevap": cevap,
            "score": float(sonuc["score"]),
            "start": int(sonuc["start"]) if "start" in sonuc else -1,
            "end": int(sonuc["end"]) if "end" in sonuc else -1,
            "model": model_name
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Özetleme API
@app.route('/api/ozetleme', methods=['POST'])
def ozetleme_api():
    data = request.json
    metin = data.get('metin', '')
    min_uzunluk = int(data.get('min_uzunluk', 30))
    max_uzunluk = int(data.get('max_uzunluk', 100))
    
    print(f"Özetleme isteği alındı: {len(metin)} karakter, min={min_uzunluk}, max={max_uzunluk}")
    
    if not metin.strip():
        return jsonify({"error": "Lütfen özetlenecek bir metin girin!"})
    
    # Metin çok kısaysa
    if len(metin.split()) < 20:
        return jsonify({"error": "Metin özetleme için çok kısa. Lütfen daha uzun bir metin girin."})
    
    try:
        # Model yükle
        model_name = "facebook/bart-large-cnn"
        
        # Önbellekten pipeline al
        ozetleme = pipeline_cache.get_pipeline("summarization", model=model_name)
        
        # Metni özetle
        ozet = ozetleme(
            metin, 
            min_length=min_uzunluk, 
            max_length=max_uzunluk,
            do_sample=False
        )
        
        # Özet metnini al
        ozet_metni = ozet[0]['summary_text']
        
        # İstatistikler
        original_length = len(metin)
        summary_length = len(ozet_metni)
        compression_ratio = 100 * (1 - (summary_length / original_length))
        
        return jsonify({
            "ozet": ozet_metni,
            "original_length": original_length,
            "summary_length": summary_length,
            "compression_ratio": compression_ratio,
            "model": model_name
        })
    
    except Exception as e:
        import traceback
        print(f"Özetleme hatası: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)})

# Çeviri API
@app.route('/api/ceviri', methods=['POST'])
def ceviri_api():
    data = request.json
    text = data.get('text', '')
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'tr')
    
    if not text.strip():
        return jsonify({"error": "Lütfen çevrilecek bir metin girin!"})
    
    try:
        # Alternatif olarak daha yaygın kullanılan bir model kullanalım
        model_name = "facebook/m2m100_418M"  # Çok dilli çeviri modeli
        
        # Önbellekten model ve tokenizer'ı al
        pipeline_key = f"translation-{source_lang}-{target_lang}"
        if pipeline_key not in pipeline_cache.pipelines:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # M2M100 modeli için özel işlem
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            
            pipeline_cache.pipelines[f"{pipeline_key}-tokenizer"] = M2M100Tokenizer.from_pretrained(model_name)
            pipeline_cache.pipelines[f"{pipeline_key}-model"] = M2M100ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        tokenizer = pipeline_cache.pipelines[f"{pipeline_key}-tokenizer"]
        model = pipeline_cache.pipelines[f"{pipeline_key}-model"]
        
        # Dil kodlarını insan tarafından okunabilir formata dönüştür
        source_lang_name = {
            "en": "İngilizce",
            "tr": "Türkçe",
            "fr": "Fransızca",
            "de": "Almanca",
            "es": "İspanyolca",
            "it": "İtalyanca",
            "ru": "Rusça",
            "zh": "Çince",
            "ja": "Japonca",
            "ar": "Arapça",
            "pt": "Portekizce",
            "ko": "Korece"
        }.get(source_lang, source_lang)
        
        target_lang_name = {
            "en": "İngilizce",
            "tr": "Türkçe",
            "fr": "Fransızca",
            "de": "Almanca",
            "es": "İspanyolca",
            "it": "İtalyanca",
            "ru": "Rusça",
            "zh": "Çince",
            "ja": "Japonca",
            "ar": "Arapça",
            "pt": "Portekizce",
            "ko": "Korece"
        }.get(target_lang, target_lang)
        
        # M2M100 modeli için çeviri işlemi
        # Kaynak dili ayarla
        tokenizer.src_lang = source_lang
        
        # Metni tokenize et
        encoded = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
            model = model.to("cuda")
        
        # Çeviriyi oluştur
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            max_length=512
        )
        
        # Çeviriyi decode et
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return jsonify({
            "translation": translation,
            "source_text": text,
            "source_lang": source_lang,
            "source_lang_name": source_lang_name,
            "target_lang": target_lang,
            "target_lang_name": target_lang_name,
            "model": model_name
        })
    
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

# NER API
@app.route('/api/ner', methods=['POST'])
def ner_api():
    data = request.json
    metin = data.get('metin', '')
    
    if not metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    try:
        # Önbellekten pipeline al - grouped_entities parametresi ile
        ner = pipeline_cache.get_pipeline("ner", grouped_entities=True)
        sonuclar = ner(metin)
        
        return jsonify({
            "sonuclar": [{
                "entity_group": sonuc['entity_group'],
                "word": sonuc['word'],
                "score": float(sonuc['score']),
                "start": int(sonuc['start']),
                "end": int(sonuc['end'])
            } for sonuc in sonuclar]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Türkçe maskeleme API
@app.route('/api/turkce-maskeleme', methods=['POST'])
def turkce_maskeleme_api():
    data = request.json
    maskeli_metin = data.get('maskeli_metin', '')
    
    if not maskeli_metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    try:
        # Türkçe BERT modeli
        model = "dbmdz/bert-base-turkish-cased"
        
        # Maske token'ı kontrol et
        if "[MASK]" not in maskeli_metin:
            return jsonify({"error": "Lütfen metinde en az bir [MASK] token'ı kullanın."})
        
        # Önbellekten pipeline al
        maskeleme = pipeline_cache.get_pipeline("fill-mask", model=model)
        sonuclar = maskeleme(maskeli_metin)
        
        return jsonify({
            "sonuclar": [{
                "sequence": sonuc['sequence'],
                "score": float(sonuc['score']),
                "token": sonuc['token_str']
            } for sonuc in sonuclar[:5]]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Metin sınıflandırma API
@app.route('/api/metin-siniflandirma', methods=['POST'])
def metin_siniflandirma_api():
    data = request.json
    metin = data.get('metin', '')
    kategori_sayisi = int(data.get('kategori_sayisi', 2))
    
    if not metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    try:
        # Model seçimi - kategori sayısına göre
        if kategori_sayisi == 2:
            model = "distilbert-base-uncased-finetuned-sst-2-english"
        else:
            model = "j-hartmann/emotion-english-distilroberta-base"  # Çok sınıflı duygu analizi
        
        # Önbellekten pipeline al
        siniflandirma = pipeline_cache.get_pipeline("text-classification", model=model)
        sonuclar = siniflandirma(metin)
        
        # Sonuçları işle
        if isinstance(sonuclar, list) and len(sonuclar) > 1:
            return jsonify({
                "sonuclar": [{
                    "label": sonuc['label'],
                    "score": float(sonuc['score'])
                } for sonuc in sonuclar]
            })
        else:
            sonuc = sonuclar[0] if isinstance(sonuclar, list) else sonuclar
            return jsonify({
                "label": sonuc['label'],
                "score": float(sonuc['score']),
                "metin": metin,
                "model": model
            })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Sıfır-atış sınıflandırma API
@app.route('/api/sifir-atis-siniflandirma', methods=['POST'])
def sifir_atis_siniflandirma_api():
    data = request.json
    metin = data.get('metin', '')
    etiketler = data.get('etiketler', '')
    
    if not metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    if not etiketler.strip():
        return jsonify({"error": "Lütfen sınıflandırma etiketlerini girin!"})
    
    try:
        # Etiketleri ayır
        etiket_listesi = [etiket.strip() for etiket in etiketler.split(",")]
        
        if len(etiket_listesi) < 2:
            return jsonify({"error": "Lütfen en az 2 etiket girin (virgülle ayırarak)!"})
        
        # Önbellekten pipeline al
        zero_shot = pipeline_cache.get_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        sonuc = zero_shot(metin, etiket_listesi, multi_label=False)
        
        return jsonify({
            "labels": sonuc['labels'],
            "scores": [float(score) for score in sonuc['scores']],
            "metin": metin,
            "etiketler": etiket_listesi
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Cümle benzerliği API
@app.route('/api/cumle-benzerligi', methods=['POST'])
def cumle_benzerligi_api():
    data = request.json
    cumle1 = data.get('cumle1', '')
    cumle2 = data.get('cumle2', '')
    
    if not cumle1.strip() or not cumle2.strip():
        return jsonify({"error": "Lütfen her iki cümleyi de girin!"})
    
    try:
        # Sentence Transformers modelini yükle
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Önbellekten kontrol et veya yeni model oluştur
        if "sentence_transformer" not in pipeline_cache.pipelines:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            pipeline_cache.pipelines["sentence_transformer"] = SentenceTransformer(model_name, device=f'cuda:{device}' if device >= 0 else 'cpu')
        
        model = pipeline_cache.pipelines["sentence_transformer"]
        
        # Cümleleri vektörlere dönüştür
        embedding1 = model.encode(cumle1, convert_to_tensor=True)
        embedding2 = model.encode(cumle2, convert_to_tensor=True)
        
        # Kosinüs benzerliğini hesapla
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        
        # Benzerlik yüzdesi
        similarity_percentage = (cosine_similarity + 1) / 2 * 100  # -1 ile 1 arasındaki değeri 0-100 aralığına dönüştür
        
        # Benzerlik seviyesi
        if similarity_percentage >= 80:
            similarity_level = "Çok Yüksek"
        elif similarity_percentage >= 60:
            similarity_level = "Yüksek"
        elif similarity_percentage >= 40:
            similarity_level = "Orta"
        elif similarity_percentage >= 20:
            similarity_level = "Düşük"
        else:
            similarity_level = "Çok Düşük"
        
        return jsonify({
            "similarity_percentage": float(similarity_percentage),
            "similarity_level": similarity_level,
            "cosine_similarity": float(cosine_similarity),
            "cumle1": cumle1,
            "cumle2": cumle2,
            "model": model_name
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Görsel soru cevaplama API
@app.route('/api/gorsel-soru-cevaplama', methods=['POST'])
def gorsel_soru_cevaplama_api():
    if 'image' not in request.files:
        return jsonify({"error": "Lütfen bir görüntü yükleyin!"})
    
    soru = request.form.get('soru', '')
    
    if not soru.strip():
        return jsonify({"error": "Lütfen bir soru girin!"})
    
    try:
        # Görüntüyü kaydet
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        
        # Görüntüyü aç
        image = Image.open(filepath)
        
        # Model yükle
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        
        # Önbellekten pipeline al
        image_to_text = pipeline_cache.get_pipeline("image-to-text", model=model_name)
        
        # Görüntüyü işle
        result = image_to_text(image)
        
        # Görüntü açıklaması
        image_caption = result[0]['generated_text']
        
        # Basit bir görsel QA simülasyonu
        # Soruyu analiz et
        soru_lower = soru.lower()
        cevap = "Görüntüyü tam olarak analiz edemiyorum."
        
        # Basit soru-cevap mantığı
        if "ne" in soru_lower or "nedir" in soru_lower:
            cevap = f"Görüntüde {image_caption.lower()} görünüyor."
        elif "var mı" in soru_lower:
            # Sorudaki anahtar kelimeleri kontrol et
            keywords = [word for word in soru_lower.split() if len(word) > 3 and word not in ["var", "mı", "bir", "bu", "şu", "ve", "ile", "için"]]
            found = any(keyword in image_caption.lower() for keyword in keywords)
            if found:
                cevap = f"Evet, görüntüde {' veya '.join(k for k in keywords if k in image_caption.lower())} var."
            else:
                cevap = f"Hayır, görüntüde bahsettiğiniz öğe görünmüyor."
        elif "kaç" in soru_lower:
            cevap = "Görüntüdeki nesnelerin sayısını tam olarak belirleyemiyorum."
        elif "nerede" in soru_lower:
            cevap = "Görüntünün tam konumunu belirleyemiyorum."
        elif "kim" in soru_lower:
            if "person" in image_caption.lower() or "man" in image_caption.lower() or "woman" in image_caption.lower():
                cevap = "Görüntüde bir kişi var, ancak kim olduğunu belirleyemiyorum."
            else:
                cevap = "Görüntüde bir kişi göremiyorum."
        else:
            cevap = f"Görüntü açıklaması: {image_caption}"
        
        # Geçici dosyayı sil
        os.remove(filepath)
        
        return jsonify({
            "cevap": cevap,
            "image_caption": image_caption,
            "soru": soru,
            "model": model_name
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Metin düzeltme API
@app.route('/api/metin-duzeltme', methods=['POST'])
def metin_duzeltme_api():
    data = request.json
    metin = data.get('metin', '')
    
    if not metin.strip():
        return jsonify({"error": "Lütfen düzeltilecek bir metin girin!"})
    
    try:
        # Model yükle
        model_name = "oliverguhr/spelling-correction-english-base"
        
        # Önbellekten pipeline al
        duzeltme = pipeline_cache.get_pipeline("text2text-generation", model=model_name)
        sonuc = duzeltme(metin, max_length=len(metin) + 50)
        
        duzeltilmis_metin = sonuc[0]['generated_text']
        
        # Değişiklikleri belirle
        from difflib import SequenceMatcher
        
        matcher = SequenceMatcher(None, metin, duzeltilmis_metin)
        changes = []
        
        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode != 'equal':
                changes.append({
                    'type': opcode,
                    'original': metin[a0:a1],
                    'corrected': duzeltilmis_metin[b0:b1],
                    'position': [a0, a1, b0, b1]
                })
        
        # Değişiklik var mı kontrol et
        if metin == duzeltilmis_metin:
            degisiklik_mesaji = "Metinde düzeltilecek bir hata bulunamadı."
        else:
            degisiklik_mesaji = "Metinde düzeltmeler yapıldı."
        
        return jsonify({
            "original_text": metin,
            "corrected_text": duzeltilmis_metin,
            "changes": changes,
            "message": degisiklik_mesaji,
            "model": model_name
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Çok sınıflı duygu analizi API
@app.route('/api/cok-sinifli-duygu-analizi', methods=['POST'])
def cok_sinifli_duygu_analizi_api():
    data = request.json
    metin = data.get('metin', '')
    
    if not metin.strip():
        return jsonify({"error": "Lütfen bir metin girin!"})
    
    try:
        # Model yükle
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        
        # Önbellekten pipeline al
        duygu_analizi = pipeline_cache.get_pipeline("text-classification", model=model_name, top_k=None)
        sonuclar = duygu_analizi(metin)
        
        # Duygu renkleri
        duygu_renkleri = {
            "joy": "#4CAF50",      # Yeşil
            "love": "#E91E63",     # Pembe
            "anger": "#F44336",    # Kırmızı
            "fear": "#FF9800",     # Turuncu
            "sadness": "#2196F3",  # Mavi
            "surprise": "#9C27B0"  # Mor
        }
        
        # Sonuçları işle
        emotions = []
        for sonuc in sonuclar[0]:
            duygu = sonuc['label']
            skor = sonuc['score']
            renk = duygu_renkleri.get(duygu, "#9E9E9E")  # Varsayılan gri
            
            emotions.append({
                "emotion": duygu,
                "score": float(skor),
                "color": renk
            })
        
        return jsonify({
            "emotions": emotions,
            "text": metin,
            "model": model_name
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Konuşma tanıma API
@app.route('/api/konusma-tanima', methods=['POST'])
def konusma_tanima_api():
    if 'audio' not in request.files:
        return jsonify({"error": "Lütfen bir ses dosyası yükleyin!"})
    
    dil = request.form.get('dil', 'tr')
    
    try:
        # Ses dosyasını kaydet
        audio_file = request.files['audio']
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # OpenAI Whisper modelini kullan
        model_name = "openai/whisper-large-v3-turbo"
        
        # Önbellekten pipeline al
        pipeline_key = f"speech-recognition-{dil}"
        if pipeline_key not in pipeline_cache.pipelines:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            pipeline_cache.pipelines[pipeline_key] = pipeline(
                "automatic-speech-recognition", 
                model=model_name,
                chunk_length_s=30,
                batch_size=16,
                device=device
            )
        
        konusma_tanima = pipeline_cache.pipelines[pipeline_key]
        
        # Dile göre generate_kwargs ayarla
        generate_kwargs = {}
        if dil == "tr":
            generate_kwargs["language"] = "turkish"
        else:  # Varsayılan İngilizce
            generate_kwargs["language"] = "english"
        
        # Ses dosyasını işle
        sonuc = konusma_tanima(
            filepath, 
            return_timestamps=True,
            generate_kwargs=generate_kwargs
        )
        
        # Dil bilgisi
        dil_adi = "Türkçe" if dil == "tr" else "İngilizce"
        
        # Zaman damgalarını işle
        timestamps = []
        if "chunks" in sonuc:
            for chunk in sonuc["chunks"]:
                start = chunk.get("timestamp", [0])[0]
                end = chunk.get("timestamp", [0, 0])[1]
                text = chunk.get("text", "")
                start_time = f"{int(start // 60)}:{int(start % 60):02d}"
                end_time = f"{int(end // 60)}:{int(end % 60):02d}"
                
                timestamps.append({
                    "start": start,
                    "end": end,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text
                })
        
        # Geçici dosyayı sil
        os.remove(filepath)
        
        return jsonify({
            "text": sonuc["text"],
            "timestamps": timestamps,
            "language": dil_adi,
            "model": model_name
        })
    
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)