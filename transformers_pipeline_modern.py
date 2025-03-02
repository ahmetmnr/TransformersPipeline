# -*- coding: utf-8 -*-
"""
Transformers Kütüphanesi - Modern Pipeline Demo Arayüzü (Optimize Edilmiş)
"""

import gradio as gr
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

# GPU kullanımı için device değişkeni
device = 0 if torch.cuda.is_available() else -1

# Daha basit CSS - sadece gerekli stiller
css = """
.result-card {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    overflow: hidden;
}
.result-header {
    padding: 0.75rem;
    color: white;
    font-weight: bold;
}
.sentiment-positive { background-color: #22c55e; }
.sentiment-negative { background-color: #ef4444; }
.sentiment-neutral { background-color: #3b82f6; }
.result-content { padding: 0.75rem; }
"""

# Daha basit tema
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue"
)

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

def duygu_analizi_demo(metin, dil="en"):
    """Duygu analizi pipeline demo fonksiyonu"""
    if not metin.strip():
        return "Lütfen bir metin girin!"
    
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
        
        # Daha basit HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-{sentiment_class.lower()}">
                {label} (Güven: {score:.4f})
            </div>
            <div class="result-content">
                <p>"{metin}"</p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def metin_uretme_demo(baslangic_metni, max_uzunluk=50, tekrar_sayisi=1):
    """Metin üretme pipeline demo fonksiyonu"""
    if not baslangic_metni.strip():
        return "Lütfen bir başlangıç metni girin!"
    
    try:
        # Önbellekten pipeline al
        metin_uretici = pipeline_cache.get_pipeline("text-generation")
        sonuclar = metin_uretici(
            baslangic_metni, 
            max_length=max_uzunluk,
            num_return_sequences=tekrar_sayisi,
            do_sample=True
        )
        
        # Daha basit HTML çıktısı
        html_output = "<div>"
        for i, sonuc in enumerate(sonuclar, 1):
            html_output += f"<div class='result-card'><div class='result-content'><strong>Sonuç {i}:</strong><p>{sonuc['generated_text']}</p></div></div>"
        html_output += "</div>"
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def maskeleme_demo(maskeli_metin):
    """Maskeleme pipeline demo fonksiyonu"""
    if not maskeli_metin.strip():
        return "Lütfen bir metin girin!"
    
    try:
        # Hangi mask token'ı kullanıldığını belirleyelim
        if "[MASK]" in maskeli_metin:
            mask_token = "[MASK]"
            model = "bert-base-uncased"
        elif "<mask>" in maskeli_metin:
            mask_token = "<mask>"
            model = "roberta-base"
        else:
            return "Lütfen maskelenmiş bir kelime ekleyin: [MASK] veya <mask>"
        
        # Önbellekten pipeline al
        maskeleme = pipeline_cache.get_pipeline("fill-mask", model=model)
        sonuclar = maskeleme(maskeli_metin)
        
        # Daha basit HTML çıktısı
        html_output = "<div>"
        for i, sonuc in enumerate(sonuclar[:5], 1):
            html_output += f"<div class='result-card'><div class='result-content'><strong>{i}.</strong> {sonuc['sequence']}<br><small>Skor: {sonuc['score']:.4f}</small></div></div>"
        html_output += "</div>"
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def soru_cevaplama_demo(soru, metin):
    """Soru cevaplama pipeline demo fonksiyonu"""
    if not soru.strip() or not metin.strip():
        return "Lütfen hem soru hem de metin girin!"
    
    try:
        # Önbellekten pipeline al
        soru_cevaplama = pipeline_cache.get_pipeline("question-answering")
        sonuc = soru_cevaplama(question=soru, context=metin)
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Cevap (Güven: {sonuc['score']:.4f})
            </div>
            <div class="result-content">
                <p><strong>Soru:</strong> {soru}</p>
                <p><strong>Cevap:</strong> {sonuc['answer']}</p>
                <p><small>Metin içinde konum: {sonuc['start']} - {sonuc['end']}</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def ozetleme_demo(uzun_metin, max_uzunluk=150, min_uzunluk=50):
    """Metin özetleme pipeline demo fonksiyonu"""
    if not uzun_metin.strip():
        return "Lütfen özetlenecek bir metin girin!"
    
    try:
        # Önbellekten pipeline al
        ozetleme = pipeline_cache.get_pipeline("summarization")
        ozet = ozetleme(
            uzun_metin, 
            max_length=max_uzunluk, 
            min_length=min_uzunluk, 
            do_sample=False
        )
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Özet
            </div>
            <div class="result-content">
                <p>{ozet[0]['summary_text']}</p>
                <hr>
                <p><small><strong>Orijinal Metin Uzunluğu:</strong> {len(uzun_metin.split())} kelime</small></p>
                <p><small><strong>Özet Uzunluğu:</strong> {len(ozet[0]['summary_text'].split())} kelime</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def ceviri_demo(text, source_lang, target_lang):
    """Çeviri pipeline demo fonksiyonu"""
    if not text.strip():
        return "Lütfen çevrilecek bir metin girin!"
    
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
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Çeviri Sonucu ({source_lang_name} → {target_lang_name})
            </div>
            <div class="result-content">
                <div style="display: flex; flex-direction: column; gap: 15px;">
                    <div>
                        <p><strong>Kaynak Metin ({source_lang_name}):</strong></p>
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px;">
                            {text}
                        </div>
                    </div>
                    <div>
                        <p><strong>Çeviri ({target_lang_name}):</strong></p>
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px;">
                            {translation}
                        </div>
                    </div>
                </div>
                <p><small><strong>Model:</strong> {model_name}</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        import traceback
        return f"Hata: {str(e)}<br><pre>{traceback.format_exc()}</pre>"

def ner_demo(metin):
    """Varlık İsmi Tanıma (NER) pipeline demo fonksiyonu"""
    if not metin.strip():
        return "Lütfen bir metin girin!"
    
    try:
        # Önbellekten pipeline al - grouped_entities parametresi ile
        ner = pipeline_cache.get_pipeline("ner", grouped_entities=True)
        sonuclar = ner(metin)
        
        # Renk sınıfları
        entity_colors = {
            "PER": "sentiment-positive",  # Kişi
            "ORG": "sentiment-negative",  # Organizasyon
            "LOC": "sentiment-neutral",   # Konum
            "MISC": "sentiment-neutral"   # Diğer
        }
        
        # HTML çıktısı
        html_output = "<div>"
        for i, sonuc in enumerate(sonuclar, 1):
            entity_type = sonuc['entity_group']
            color_class = entity_colors.get(entity_type, "sentiment-neutral")
            
            html_output += f"""<div class="result-card">
                <div class="result-header {color_class}">
                    {entity_type} (Güven: {sonuc['score']:.4f})
                </div>
                <div class="result-content">
                    <p><strong>Metin:</strong> {sonuc['word']}</p>
                    <p><small>Konum: {sonuc['start']} - {sonuc['end']}</small></p>
                </div>
            </div>"""
        
        if not sonuclar:
            html_output += "<p>Metinde tanımlanabilir varlık bulunamadı.</p>"
        
        html_output += "</div>"
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def turkce_maskeleme_demo(maskeli_metin):
    """Türkçe maskeleme pipeline demo fonksiyonu"""
    if not maskeli_metin.strip():
        return "Lütfen bir metin girin!"
    
    try:
        # Türkçe BERT modeli
        model = "dbmdz/bert-base-turkish-cased"
        
        # Maske token'ı kontrol et
        if "[MASK]" not in maskeli_metin:
            return "Lütfen metinde en az bir [MASK] token'ı kullanın."
        
        # Önbellekten pipeline al
        maskeleme = pipeline_cache.get_pipeline("fill-mask", model=model)
        sonuclar = maskeleme(maskeli_metin)
        
        # HTML çıktısı
        html_output = "<div>"
        for i, sonuc in enumerate(sonuclar[:5], 1):
            html_output += f"""<div class="result-card">
                <div class="result-content">
                    <strong>{i}.</strong> {sonuc['sequence']}
                    <br><small>Skor: {sonuc['score']:.4f}</small>
                </div>
            </div>"""
        
        html_output += "</div>"
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def metin_siniflandirma_demo(metin, kategori_sayisi=2):
    """Metin sınıflandırma pipeline demo fonksiyonu"""
    if not metin.strip():
        return "Lütfen bir metin girin!"
    
    try:
        # Model seçimi - kategori sayısına göre
        if kategori_sayisi == 2:
            model = "distilbert-base-uncased-finetuned-sst-2-english"
        else:
            model = "j-hartmann/emotion-english-distilroberta-base"  # Çok sınıflı duygu analizi
        
        # Önbellekten pipeline al
        siniflandirma = pipeline_cache.get_pipeline("text-classification", model=model)
        sonuclar = siniflandirma(metin)
        
        # Birden fazla sonuç varsa (top_k parametresi kullanıldıysa)
        if isinstance(sonuclar, list) and len(sonuclar) > 1:
            # HTML çıktısı - birden fazla sonuç
            html_output = "<div>"
            for i, sonuc in enumerate(sonuclar, 1):
                label = sonuc['label']
                score = sonuc['score']
                
                # Renk sınıfı belirle
                if "POSITIVE" in label or "joy" in label or "love" in label:
                    color_class = "sentiment-positive"
                elif "NEGATIVE" in label or "anger" in label or "sadness" in label:
                    color_class = "sentiment-negative"
                else:
                    color_class = "sentiment-neutral"
                
                html_output += f"""<div class="result-card">
                    <div class="result-header {color_class}">
                        {label} (Güven: {score:.4f})
                    </div>
                    <div class="result-content">
                        <p>Sıralama: {i}</p>
                    </div>
                </div>"""
            
            html_output += "</div>"
        else:
            # Tek sonuç
            sonuc = sonuclar[0] if isinstance(sonuclar, list) else sonuclar
            label = sonuc['label']
            score = sonuc['score']
            
            # Renk sınıfı belirle
            if "POSITIVE" in label or "joy" in label or "love" in label:
                color_class = "sentiment-positive"
            elif "NEGATIVE" in label or "anger" in label or "sadness" in label:
                color_class = "sentiment-negative"
            else:
                color_class = "sentiment-neutral"
            
            html_output = f"""<div class="result-card">
                <div class="result-header {color_class}">
                    {label} (Güven: {score:.4f})
                </div>
                <div class="result-content">
                    <p>Metin: "{metin}"</p>
                    <p>Model: {model}</p>
                </div>
            </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def sifir_atis_siniflandirma_demo(metin, etiketler):
    """Sıfır-atış sınıflandırma pipeline demo fonksiyonu"""
    if not metin.strip():
        return "Lütfen bir metin girin!"
    
    if not etiketler.strip():
        return "Lütfen sınıflandırma etiketlerini girin!"
    
    try:
        # Etiketleri ayır
        etiket_listesi = [etiket.strip() for etiket in etiketler.split(",")]
        
        if len(etiket_listesi) < 2:
            return "Lütfen en az 2 etiket girin (virgülle ayırarak)!"
        
        # Önbellekten pipeline al
        zero_shot = pipeline_cache.get_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        sonuc = zero_shot(metin, etiket_listesi, multi_label=False)
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Sıfır-Atış Sınıflandırma Sonuçları
            </div>
            <div class="result-content">
                <p><strong>Metin:</strong> {metin}</p>
                <p><strong>Etiketler:</strong> {', '.join(etiket_listesi)}</p>
                <hr>
                <p><strong>Sonuçlar:</strong></p>
                <ul>"""
        
        # Etiketleri ve skorları listele
        for i, (etiket, skor) in enumerate(zip(sonuc['labels'], sonuc['scores']), 1):
            # Renk sınıfı belirle - en yüksek skora sahip etiket için farklı renk
            color_class = "sentiment-positive" if i == 1 else ""
            html_output += f'<li class="{color_class}"><strong>{etiket}</strong>: {skor:.4f}</li>'
        
        html_output += """</ul>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def cumle_benzerligi_demo(cumle1, cumle2):
    """Cümle benzerliği demo fonksiyonu"""
    if not cumle1.strip() or not cumle2.strip():
        return "Lütfen her iki cümleyi de girin!"
    
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
            color_class = "sentiment-positive"
        elif similarity_percentage >= 60:
            similarity_level = "Yüksek"
            color_class = "sentiment-positive"
        elif similarity_percentage >= 40:
            similarity_level = "Orta"
            color_class = "sentiment-neutral"
        elif similarity_percentage >= 20:
            similarity_level = "Düşük"
            color_class = "sentiment-negative"
        else:
            similarity_level = "Çok Düşük"
            color_class = "sentiment-negative"
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header {color_class}">
                Benzerlik: {similarity_percentage:.2f}% ({similarity_level})
            </div>
            <div class="result-content">
                <p><strong>Cümle 1:</strong> {cumle1}</p>
                <p><strong>Cümle 2:</strong> {cumle2}</p>
                <p><strong>Model:</strong> {model_name}</p>
                <p><small>Kosinüs Benzerliği: {cosine_similarity:.4f}</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def gorsel_soru_cevaplama_demo(image, soru):
    """Görsel soru cevaplama demo fonksiyonu"""
    if image is None:
        return "Lütfen bir görüntü yükleyin!"
    
    if not soru.strip():
        return "Lütfen bir soru girin!"
    
    try:
        # Model yükle
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        
        # Önbellekten pipeline al
        image_to_text = pipeline_cache.get_pipeline("image-to-text", model=model_name)
        
        # Görüntüyü işle
        result = image_to_text(image)
        
        # Görüntü açıklaması
        image_caption = result[0]['generated_text']
        
        # Basit bir görsel QA simülasyonu
        # Not: Bu gerçek bir VQA modeli değil, sadece görüntü açıklaması ve soruyu birleştiriyor
        # Gerçek bir VQA modeli için daha karmaşık bir yapı gerekir
        
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
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Görsel Analiz Sonucu
            </div>
            <div class="result-content">
                <p><strong>Soru:</strong> {soru}</p>
                <p><strong>Cevap:</strong> {cevap}</p>
                <hr>
                <p><small><strong>Görüntü Açıklaması:</strong> {image_caption}</small></p>
                <p><small><strong>Model:</strong> {model_name}</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def metin_duzeltme_demo(metin):
    """Metin düzeltme pipeline demo fonksiyonu"""
    if not metin.strip():
        return "Lütfen düzeltilecek bir metin girin!"
    
    try:
        # Model yükle
        model_name = "oliverguhr/spelling-correction-english-base"
        
        # Önbellekten pipeline al
        duzeltme = pipeline_cache.get_pipeline("text2text-generation", model=model_name)
        sonuc = duzeltme(metin, max_length=len(metin) + 50)
        
        duzeltilmis_metin = sonuc[0]['generated_text']
        
        # Değişiklikleri vurgula
        from difflib import SequenceMatcher
        
        def highlight_diff(a, b):
            matcher = SequenceMatcher(None, a, b)
            highlighted = ""
            for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
                if opcode == 'equal':
                    highlighted += a[a0:a1]
                elif opcode == 'insert':
                    highlighted += f'<span style="background-color: #c8e6c9; font-weight: bold;">{b[b0:b1]}</span>'
                elif opcode == 'delete':
                    highlighted += f'<span style="background-color: #ffcdd2; text-decoration: line-through;">{a[a0:a1]}</span>'
                elif opcode == 'replace':
                    highlighted += f'<span style="background-color: #ffcdd2; text-decoration: line-through;">{a[a0:a1]}</span>'
                    highlighted += f'<span style="background-color: #c8e6c9; font-weight: bold;">{b[b0:b1]}</span>'
            return highlighted
        
        # Değişiklik var mı kontrol et
        if metin == duzeltilmis_metin:
            degisiklik_mesaji = "Metinde düzeltilecek bir hata bulunamadı."
            vurgulu_metin = metin
        else:
            degisiklik_mesaji = "Metinde düzeltmeler yapıldı."
            vurgulu_metin = highlight_diff(metin, duzeltilmis_metin)
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Metin Düzeltme Sonucu
            </div>
            <div class="result-content">
                <p><strong>Orijinal Metin:</strong> {metin}</p>
                <p><strong>Düzeltilmiş Metin:</strong> {duzeltilmis_metin}</p>
                <hr>
                <p><strong>Değişiklikler:</strong> {degisiklik_mesaji}</p>
                <p>{vurgulu_metin}</p>
                <p><small><strong>Model:</strong> {model_name}</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def cok_sinifli_duygu_analizi_demo(metin):
    """Çok sınıflı duygu analizi pipeline demo fonksiyonu"""
    if not metin.strip():
        return "Lütfen bir metin girin!"
    
    try:
        # Model yükle
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        
        # Önbellekten pipeline al
        duygu_analizi = pipeline_cache.get_pipeline("text-classification", model=model, top_k=None)
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
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Çok Sınıflı Duygu Analizi
            </div>
            <div class="result-content">
                <p><strong>Metin:</strong> {metin}</p>
                <hr>
                <p><strong>Duygular:</strong></p>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">"""
        
        # Duygu çubukları
        for sonuc in sonuclar[0]:
            duygu = sonuc['label']
            skor = sonuc['score']
            renk = duygu_renkleri.get(duygu, "#9E9E9E")  # Varsayılan gri
            
            html_output += f"""
                <div style="flex: 1; min-width: 150px;">
                    <div style="margin-bottom: 5px;"><strong>{duygu.capitalize()}</strong>: {skor:.4f}</div>
                    <div style="background-color: #f0f0f0; border-radius: 4px; height: 20px; width: 100%;">
                        <div style="background-color: {renk}; height: 100%; width: {skor * 100}%; border-radius: 4px;"></div>
                    </div>
                </div>"""
        
        html_output += """</div>
                <p><small><strong>Model:</strong> j-hartmann/emotion-english-distilroberta-base</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        return f"Hata: {str(e)}"

def konusma_tanima_demo(audio, dil):
    """Konuşma tanıma pipeline demo fonksiyonu"""
    if audio is None:
        return "Lütfen bir ses dosyası yükleyin!"
    
    try:
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
            audio, 
            return_timestamps=True,
            generate_kwargs=generate_kwargs  # task ve language parametrelerini generate_kwargs içinde geçir
        )
        
        # Dil bilgisi
        dil_adi = "Türkçe" if dil == "tr" else "İngilizce"
        
        # Zaman damgalarını işle
        timestamps_html = ""
        if "chunks" in sonuc:
            timestamps_html = "<p><strong>Zaman Damgaları:</strong></p><ul>"
            for chunk in sonuc["chunks"]:
                start = chunk.get("timestamp", [0])[0]
                end = chunk.get("timestamp", [0, 0])[1]
                text = chunk.get("text", "")
                start_time = f"{int(start // 60)}:{int(start % 60):02d}"
                end_time = f"{int(end // 60)}:{int(end % 60):02d}"
                timestamps_html += f"<li><strong>{start_time} - {end_time}:</strong> {text}</li>"
            timestamps_html += "</ul>"
        
        # HTML çıktısı
        html_output = f"""<div class="result-card">
            <div class="result-header sentiment-neutral">
                Konuşma Tanıma Sonucu ({dil_adi})
            </div>
            <div class="result-content">
                <p><strong>Transkripsiyon:</strong> {sonuc["text"]}</p>
                {timestamps_html}
                <p><small><strong>Model:</strong> {model_name}</small></p>
            </div>
        </div>"""
        
        return html_output
    
    except Exception as e:
        import traceback
        return f"Hata: {str(e)}<br><pre>{traceback.format_exc()}</pre>"

# Arayüz oluşturma
with gr.Blocks(title="Transformers Pipeline Demo", css=css, theme=theme) as demo:
    gr.Markdown("# 🤗 Transformers Pipeline Demo")
    gr.Markdown("HuggingFace Transformers kütüphanesi ile modern NLP uygulamaları")
    
    # GPU bilgisi
    if torch.cuda.is_available():
        gpu_info = f"🚀 GPU: {torch.cuda.get_device_name(0)}"
    else:
        gpu_info = "⚠️ GPU bulunamadı. İşlemler CPU üzerinde çalışacak."
    
    gr.Markdown(f"### {gpu_info}")
    
    with gr.Tabs():
        with gr.TabItem("😊 Duygu Analizi"):
            with gr.Row():
                with gr.Column():
                    dil = gr.Radio(
                        ["en", "tr"], 
                        label="Dil", 
                        value="en"
                    )
                    metin_input = gr.Textbox(
                        label="Metni Girin", 
                        placeholder="Analiz edilecek metni buraya yazın...",
                        value="I really enjoyed this movie, it was fantastic!",
                        lines=3
                    )
                    duygu_button = gr.Button("Duygu Analizi Yap", variant="primary")
                
                with gr.Column():
                    duygu_output = gr.HTML(label="Sonuç")
            
            duygu_button.click(
                duygu_analizi_demo, 
                inputs=[metin_input, dil], 
                outputs=duygu_output
            )
        
        with gr.TabItem("✍️ Metin Üretme"):
            with gr.Row():
                with gr.Column():
                    baslangic_input = gr.Textbox(
                        label="Başlangıç Metni", 
                        placeholder="Metin üretmek için başlangıç metni girin...",
                        value="Artificial intelligence will",
                        lines=3
                    )
                    
                    with gr.Row():
                        max_uzunluk = gr.Slider(
                            minimum=10, maximum=100, value=50, 
                            label="Maksimum Uzunluk"
                        )
                        tekrar_sayisi = gr.Slider(
                            minimum=1, maximum=3, value=2, step=1,
                            label="Üretilecek Metin Sayısı"
                        )
                    
                    uretme_button = gr.Button("Metin Üret", variant="primary")
                
                with gr.Column():
                    uretilen_output = gr.HTML(label="Üretilen Metin")
            
            uretme_button.click(
                metin_uretme_demo, 
                inputs=[baslangic_input, max_uzunluk, tekrar_sayisi], 
                outputs=uretilen_output
            )
        
        with gr.TabItem("🎭 Maskeleme"):
            with gr.Row():
                with gr.Column():
                    mask_input = gr.Textbox(
                        label="Maskelenmiş Metin", 
                        placeholder="Maskeli metni buraya yazın...",
                        value="HuggingFace is creating a <mask> that the community uses to solve NLP tasks.",
                        lines=3
                    )
                    mask_button = gr.Button("Maskeyi Doldur", variant="primary")
                
                with gr.Column():
                    mask_output = gr.HTML(label="Sonuçlar")
            
            mask_button.click(maskeleme_demo, inputs=mask_input, outputs=mask_output)
        
        with gr.TabItem("❓ Soru Cevaplama"):
            with gr.Row():
                with gr.Column():
                    soru_input = gr.Textbox(
                        label="Soru", 
                        placeholder="Sorunuzu buraya yazın...",
                        value="HuggingFace nedir?",
                        lines=2
                    )
                    metin_input = gr.Textbox(
                        label="Metin", 
                        placeholder="Sorunun cevaplanacağı metni buraya yazın...",
                        value="HuggingFace, doğal dil işleme alanında kullanılan açık kaynaklı kütüphaneler ve modeller geliştiren bir şirkettir. Transformers kütüphanesi ile tanınır ve yapay zeka topluluğuna katkıda bulunur.",
                        lines=6
                    )
                    soru_button = gr.Button("Soruyu Cevapla", variant="primary")
                
                with gr.Column():
                    soru_output = gr.HTML(label="Cevap")
            
            soru_button.click(
                soru_cevaplama_demo, 
                inputs=[soru_input, metin_input], 
                outputs=soru_output
            )

        with gr.TabItem("📝 Özetleme"):
            with gr.Row():
                with gr.Column():
                    uzun_metin_input = gr.Textbox(
                        label="Uzun Metin", 
                        placeholder="Özetlenecek metni buraya yazın...",
                        value="Yapay zeka (YZ), insan zekasını taklit eden ve topladıkları bilgilere göre yinelemeli olarak kendilerini iyileştirebilen sistemler veya makineler anlamına gelir. YZ, dar (veya zayıf) YZ ve genel YZ olarak ikiye ayrılır. Dar YZ, belirli görevleri yerine getirmek için tasarlanmıştır ve günümüzde yaygın olarak kullanılmaktadır. Genel YZ ise henüz tam olarak geliştirilmemiştir ve insan benzeri düşünme ve öğrenme yeteneğine sahip olacaktır. YZ, doğal dil işleme, bilgisayarlı görü, robotik, makine öğrenimi ve derin öğrenme gibi çeşitli alt alanları içerir.",
                        lines=8
                    )
                    
                    with gr.Row():
                        min_uzunluk = gr.Slider(
                            minimum=10, maximum=100, value=30, 
                            label="Minimum Özet Uzunluğu"
                        )
                        max_uzunluk = gr.Slider(
                            minimum=50, maximum=200, value=100, 
                            label="Maksimum Özet Uzunluğu"
                        )
                    
                    ozet_button = gr.Button("Özetle", variant="primary")
                
                with gr.Column():
                    ozet_output = gr.HTML(label="Özet")
            
            ozet_button.click(
                ozetleme_demo, 
                inputs=[uzun_metin_input, max_uzunluk, min_uzunluk], 
                outputs=ozet_output
            )

        with gr.TabItem("🌐 Çeviri"):
            with gr.Row():
                with gr.Column():
                    ceviri_input = gr.Textbox(
                        label="Çevrilecek Metin", 
                        placeholder="Çevrilecek metni buraya yazın...",
                        value="HuggingFace is a company that provides tools for natural language processing.",
                        lines=3
                    )
                    
                    with gr.Row():
                        kaynak_dil = gr.Dropdown(
                            ["en", "fr", "de", "es"], 
                            label="Kaynak Dil", 
                            value="en"
                        )
                        hedef_dil = gr.Dropdown(
                            ["tr", "fr", "de", "es", "ru"], 
                            label="Hedef Dil", 
                            value="tr"
                        )
                    
                    ceviri_button = gr.Button("Çevir", variant="primary")
                
                with gr.Column():
                    ceviri_output = gr.HTML(label="Çeviri")
            
            ceviri_button.click(
                ceviri_demo, 
                inputs=[ceviri_input, kaynak_dil, hedef_dil], 
                outputs=ceviri_output
            )

        with gr.TabItem("🏷️ Varlık İsmi Tanıma (NER)"):
            with gr.Row():
                with gr.Column():
                    ner_input = gr.Textbox(
                        label="Metin", 
                        placeholder="Metni buraya yazın...",
                        value="Apple Inc. is planning to open a new store in Istanbul, Turkiye next year. CEO Tim Cook announced this during his visit to Berlin, Germany.",
                        lines=3
                    )
                    ner_button = gr.Button("Varlıkları Tanı", variant="primary")
                
                with gr.Column():
                    ner_output = gr.HTML(label="Bulunan Varlıklar")
            
            ner_button.click(ner_demo, inputs=ner_input, outputs=ner_output)

        with gr.TabItem("🇹🇷 Türkçe Maskeleme"):
            with gr.Row():
                with gr.Column():
                    turkce_mask_input = gr.Textbox(
                        label="Maskelenmiş Türkçe Metin", 
                        placeholder="Maskeli metni buraya yazın...",
                        value="Yapay zeka [MASK] alanında devrim yaratıyor.",
                        lines=3
                    )
                    turkce_mask_button = gr.Button("Maskeyi Doldur", variant="primary")
                
                with gr.Column():
                    turkce_mask_output = gr.HTML(label="Sonuçlar")
            
            turkce_mask_button.click(turkce_maskeleme_demo, inputs=turkce_mask_input, outputs=turkce_mask_output)

        with gr.TabItem("📋 Metin Sınıflandırma"):
            with gr.Row():
                with gr.Column():
                    metin_input = gr.Textbox(
                        label="Metin", 
                        placeholder="Metin girin...",
                        value="I really enjoyed this movie, it was fantastic!",
                        lines=3
                    )
                    kategori_sayisi = gr.Slider(
                        minimum=2, maximum=5, value=2,
                        label="Kategori Sayısı"
                    )
                    siniflandirma_button = gr.Button("Metin Sınıflandır", variant="primary")
                
                with gr.Column():
                    siniflandirma_output = gr.HTML(label="Sonuç")
            
            siniflandirma_button.click(
                metin_siniflandirma_demo, 
                inputs=[metin_input, kategori_sayisi], 
                outputs=siniflandirma_output
            )

        with gr.TabItem("📋 Sıfır-Atış Sınıflandırma"):
            with gr.Row():
                with gr.Column():
                    metin_input = gr.Textbox(
                        label="Metin", 
                        placeholder="Metin girin...",
                        value="I really enjoyed this movie, it was fantastic!",
                        lines=3
                    )
                    etiketler_input = gr.Textbox(
                        label="Etiketler", 
                        placeholder="Etiketleri virgülle ayırarak girin...",
                        value="positive,negative,neutral",
                        lines=3
                    )
                    sifir_atis_button = gr.Button("Sıfır-Atış Sınıflandır", variant="primary")
                
                with gr.Column():
                    sifir_atis_output = gr.HTML(label="Sonuç")
            
            sifir_atis_button.click(
                sifir_atis_siniflandirma_demo, 
                inputs=[metin_input, etiketler_input], 
                outputs=sifir_atis_output
            )

        with gr.TabItem("🔄 Cümle Benzerliği"):
            with gr.Row():
                with gr.Column():
                    cumle1_input = gr.Textbox(
                        label="Cümle 1", 
                        placeholder="İlk cümleyi buraya yazın...",
                        value="Yapay zeka, insan zekasını taklit eden sistemlerdir.",
                        lines=3
                    )
                    
                    cumle2_input = gr.Textbox(
                        label="Cümle 2", 
                        placeholder="İkinci cümleyi buraya yazın...",
                        value="AI, insan benzeri zeka gösteren bilgisayar sistemleridir.",
                        lines=3
                    )
                    
                    benzerlik_button = gr.Button("Benzerliği Hesapla", variant="primary")
                
                with gr.Column():
                    benzerlik_output = gr.HTML(label="Benzerlik Sonucu")
            
            benzerlik_button.click(
                cumle_benzerligi_demo, 
                inputs=[cumle1_input, cumle2_input], 
                outputs=benzerlik_output
            )

        with gr.TabItem("🖼️ Görsel Soru Cevaplama"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Görüntü", 
                        type="pil",
                        sources=["upload", "clipboard"]
                    )
                    
                    soru_input = gr.Textbox(
                        label="Soru", 
                        placeholder="Görüntü hakkında bir soru sorun...",
                        value="Bu görüntüde ne var?",
                        lines=2
                    )
                    
                    gorsel_button = gr.Button("Soruyu Cevapla", variant="primary")
                
                with gr.Column():
                    gorsel_output = gr.HTML(label="Cevap")
            
            gorsel_button.click(
                gorsel_soru_cevaplama_demo, 
                inputs=[image_input, soru_input], 
                outputs=gorsel_output
            )

        with gr.TabItem("📝 Metin Düzeltme"):
            with gr.Row():
                with gr.Column():
                    duzeltme_input = gr.Textbox(
                        label="Düzeltilecek Metin", 
                        placeholder="Düzeltilecek metni buraya yazın...",
                        value="I havv a problm with my computr. It dosnt work proprly.",
                        lines=3
                    )
                    duzeltme_button = gr.Button("Metni Düzelt", variant="primary")
                
                with gr.Column():
                    duzeltme_output = gr.HTML(label="Düzeltilmiş Metin")
            
            duzeltme_button.click(
                metin_duzeltme_demo, 
                inputs=duzeltme_input, 
                outputs=duzeltme_output
            )

        with gr.TabItem("😊 Çok Sınıflı Duygu Analizi"):
            with gr.Row():
                with gr.Column():
                    cok_duygu_input = gr.Textbox(
                        label="Metin", 
                        placeholder="Analiz edilecek metni buraya yazın...",
                        value="I'm so happy to see you again! It's been a long time and I missed you so much.",
                        lines=3
                    )
                    cok_duygu_button = gr.Button("Duyguları Analiz Et", variant="primary")
                
                with gr.Column():
                    cok_duygu_output = gr.HTML(label="Duygu Analizi Sonucu")
            
            cok_duygu_button.click(
                cok_sinifli_duygu_analizi_demo, 
                inputs=cok_duygu_input, 
                outputs=cok_duygu_output
            )

        with gr.TabItem("🎤 Konuşma Tanıma"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Ses Dosyası", 
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    
                    konusma_dil = gr.Radio(
                        choices=["en", "tr"], 
                        label="Konuşma Dili", 
                        value="tr",  # Varsayılan olarak Türkçe seçili
                        info="en: İngilizce, tr: Türkçe"
                    )
                    
                    konusma_button = gr.Button("Konuşmayı Tanı", variant="primary")
                
                with gr.Column():
                    konusma_output = gr.HTML(label="Transkripsiyon")
            
            konusma_button.click(
                fn=konusma_tanima_demo, 
                inputs=[audio_input, konusma_dil], 
                outputs=konusma_output
            )
    
    # Basit footer
    gr.Markdown("---\n*🤗 Transformers Pipeline Demo | Hugging Face Türkçe Eğitim*")

# Demo'yu çalıştır
if __name__ == "__main__":
    demo.launch(share=True) 