# -*- coding: utf-8 -*-
"""
Transformers Kütüphanesi - Pipeline Kullanımı Demo
"""

import gradio as gr
from transformers import pipeline
import torch

# GPU kullanımı için device değişkeni
device = 0 if torch.cuda.is_available() else -1

def duygu_analizi_demo(metin, dil="en"):
    """Duygu analizi pipeline demo fonksiyonu"""
    if dil == "en":
        model = "distilbert-base-uncased-finetuned-sst-2-english"
    else:
        model = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    duygu_analizi = pipeline("sentiment-analysis", model=model, device=device)
    sonuc = duygu_analizi(metin)
    
    # Sonucu formatlayalım
    label = sonuc[0]["label"]
    score = sonuc[0]["score"]
    
    return f"Duygu: {label}", f"Güven Skoru: {score:.4f}"

def metin_uretme_demo(baslangic_metni, max_uzunluk=50, tekrar_sayisi=1):
    """Metin üretme pipeline demo fonksiyonu"""
    metin_uretici = pipeline("text-generation", device=device)
    sonuclar = metin_uretici(
        baslangic_metni, 
        max_length=max_uzunluk,
        num_return_sequences=tekrar_sayisi
    )
    
    uretilen_metinler = ""
    for i, sonuc in enumerate(sonuclar, 1):
        uretilen_metinler += f"Sonuç {i}:\n{sonuc['generated_text']}\n\n"
    
    return uretilen_metinler

def maskeleme_demo(maskeli_metin):
    """Maskeleme pipeline demo fonksiyonu"""
    maskeleme = pipeline("fill-mask", device=device)
    
    # Hangi mask token'ı kullanıldığını belirleyelim
    if "[MASK]" in maskeli_metin:
        mask_token = "[MASK]"
        model = "bert-base-uncased"
    elif "<mask>" in maskeli_metin:
        mask_token = "<mask>"
        model = "roberta-base"
    else:
        return "Lütfen maskelenmiş bir kelime ekleyin: [MASK] veya <mask>"
    
    maskeleme = pipeline("fill-mask", model=model, device=device)
    sonuclar = maskeleme(maskeli_metin)
    
    cikti = ""
    for i, sonuc in enumerate(sonuclar[:5], 1):
        cikti += f"{i}. {sonuc['sequence']}\n   (Skor: {sonuc['score']:.4f})\n\n"
    
    return cikti

def soru_cevap_demo(soru, metin):
    """Soru cevaplama pipeline demo fonksiyonu"""
    soru_cevaplama = pipeline("question-answering", device=device)
    sonuc = soru_cevaplama(question=soru, context=metin)
    
    return (
        f"Cevap: {sonuc['answer']}\n\n"
        f"Güven Skoru: {sonuc['score']:.4f}\n"
        f"Başlangıç: {sonuc['start']}\n"
        f"Bitiş: {sonuc['end']}"
    )

def ozetleme_demo(uzun_metin, max_uzunluk=150, min_uzunluk=50):
    """Metin özetleme pipeline demo fonksiyonu"""
    ozetleme = pipeline("summarization", device=device)
    ozet = ozetleme(
        uzun_metin, 
        max_length=max_uzunluk, 
        min_length=min_uzunluk, 
        do_sample=False
    )
    
    return ozet[0]['summary_text']

def ceviri_demo(metin, kaynak_dil="en", hedef_dil="tr"):
    """Çeviri pipeline demo fonksiyonu"""
    model_adi = f"Helsinki-NLP/opus-mt-{kaynak_dil}-{hedef_dil}"
    
    try:
        ceviri = pipeline("translation", model=model_adi, device=device)
        sonuc = ceviri(metin)
        return sonuc[0]['translation_text']
    except Exception as e:
        return f"Hata: {e}\n\nBu dil çifti için model bulunamadı. Lütfen geçerli bir dil çifti seçin."

def ner_demo(metin):
    """Varlık ismi tanıma pipeline demo fonksiyonu"""
    ner = pipeline("ner", grouped_entities=True, device=device)
    sonuclar = ner(metin)
    
    cikti = ""
    for sonuc in sonuclar:
        cikti += f"• {sonuc['word']} ({sonuc['entity_group']})\n"
        cikti += f"  Güven: {sonuc['score']:.4f}, Pozisyon: {sonuc['start']}-{sonuc['end']}\n\n"
    
    return cikti

def turkce_maskeleme_demo(maskeli_metin):
    """Türkçe maskeleme pipeline demo fonksiyonu"""
    try:
        turkce_maskeleme = pipeline(
            "fill-mask", 
            model="dbmdz/bert-base-turkish-cased", 
            device=device
        )
        
        if "[MASK]" not in maskeli_metin:
            return "Lütfen [MASK] token'ı ekleyin. Örnek: 'Türkiye'nin başkenti [MASK] şehridir.'"
        
        sonuclar = turkce_maskeleme(maskeli_metin)
        
        cikti = ""
        for i, sonuc in enumerate(sonuclar[:5], 1):
            cikti += f"{i}. {sonuc['sequence']}\n   (Skor: {sonuc['score']:.4f})\n\n"
        
        return cikti
    except Exception as e:
        return f"Hata: {e}\n\nTürkçe model yüklenirken bir sorun oluştu."

# Arayüz oluşturma
with gr.Blocks(title="Transformers Pipeline Demo") as demo:
    gr.Markdown("# 🤗 Transformers Pipeline Demo")
    gr.Markdown("Bu demo, Hugging Face Transformers kütüphanesindeki pipeline fonksiyonlarını göstermektedir.")
    
    with gr.Tab("Duygu Analizi"):
        with gr.Row():
            with gr.Column():
                dil = gr.Radio(["en", "tr"], label="Dil", value="en")
                metin_input = gr.Textbox(
                    label="Metni Girin", 
                    placeholder="Metin buraya yazın...",
                    value="I really enjoyed this movie, it was fantastic!"
                )
                duygu_button = gr.Button("Duygu Analizi Yap")
            
            with gr.Column():
                duygu_output = gr.Textbox(label="Duygu")
                skor_output = gr.Textbox(label="Skor")
        
        duygu_button.click(
            duygu_analizi_demo, 
            inputs=[metin_input, dil], 
            outputs=[duygu_output, skor_output]
        )
    
    with gr.Tab("Metin Üretme"):
        with gr.Row():
            with gr.Column():
                baslangic_input = gr.Textbox(
                    label="Başlangıç Metni", 
                    placeholder="Başlangıç metni buraya yazın...",
                    value="Artificial intelligence will"
                )
                max_uzunluk = gr.Slider(
                    minimum=10, maximum=100, value=50, 
                    label="Maksimum Uzunluk"
                )
                tekrar_sayisi = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="Üretilecek Metin Sayısı"
                )
                uretme_button = gr.Button("Metin Üret")
            
            with gr.Column():
                uretilen_output = gr.Textbox(label="Üretilen Metin", lines=10)
        
        uretme_button.click(
            metin_uretme_demo, 
            inputs=[baslangic_input, max_uzunluk, tekrar_sayisi], 
            outputs=uretilen_output
        )
    
    with gr.Tab("Maskeleme"):
        with gr.Row():
            with gr.Column():
                mask_input = gr.Textbox(
                    label="Maskelenmiş Metin", 
                    placeholder="Maskeli metni buraya yazın...",
                    value="HuggingFace is creating a <mask> that the community uses to solve NLP tasks."
                )
                mask_button = gr.Button("Maskeyi Doldur")
            
            with gr.Column():
                mask_output = gr.Textbox(label="Sonuçlar", lines=10)
        
        mask_button.click(maskeleme_demo, inputs=mask_input, outputs=mask_output)
    
    with gr.Tab("Türkçe Maskeleme"):
        with gr.Row():
            with gr.Column():
                tr_mask_input = gr.Textbox(
                    label="Maskelenmiş Türkçe Metin", 
                    placeholder="Maskeli metni buraya yazın...",
                    value="Türkiye'nin başkenti [MASK] şehridir."
                )
                tr_mask_button = gr.Button("Maskeyi Doldur")
            
            with gr.Column():
                tr_mask_output = gr.Textbox(label="Sonuçlar", lines=10)
        
        tr_mask_button.click(turkce_maskeleme_demo, inputs=tr_mask_input, outputs=tr_mask_output)
    
    with gr.Tab("Soru Cevaplama"):
        with gr.Row():
            with gr.Column():
                soru_input = gr.Textbox(
                    label="Soru", 
                    placeholder="Sorunuzu buraya yazın...",
                    value="What is HuggingFace?"
                )
                metin_input = gr.Textbox(
                    label="Metin", 
                    placeholder="Metni buraya yazın...",
                    value="HuggingFace is an AI company that develops tools for building applications using machine learning. It is known for its Transformers library.",
                    lines=5
                )
                soru_button = gr.Button("Cevapla")
            
            with gr.Column():
                cevap_output = gr.Textbox(label="Cevap", lines=5)
        
        soru_button.click(
            soru_cevap_demo, 
            inputs=[soru_input, metin_input], 
            outputs=cevap_output
        )
    
    with gr.Tab("Özetleme"):
        with gr.Row():
            with gr.Column():
                uzun_metin_input = gr.Textbox(
                    label="Uzun Metin", 
                    placeholder="Özetlenecek metni buraya yazın...",
                    value="Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term 'artificial intelligence' is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.",
                    lines=8
                )
                max_ozet = gr.Slider(
                    minimum=30, maximum=200, value=100, 
                    label="Maksimum Özet Uzunluğu"
                )
                min_ozet = gr.Slider(
                    minimum=10, maximum=100, value=30, 
                    label="Minimum Özet Uzunluğu"
                )
                ozet_button = gr.Button("Özetle")
            
            with gr.Column():
                ozet_output = gr.Textbox(label="Özet", lines=5)
        
        ozet_button.click(
            ozetleme_demo, 
            inputs=[uzun_metin_input, max_ozet, min_ozet], 
            outputs=ozet_output
        )
    
    with gr.Tab("Çeviri"):
        with gr.Row():
            with gr.Column():
                ceviri_input = gr.Textbox(
                    label="Çevrilecek Metin", 
                    placeholder="Çevrilecek metni buraya yazın...",
                    value="HuggingFace is a company that provides tools for natural language processing."
                )
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
                ceviri_button = gr.Button("Çevir")
            
            with gr.Column():
                ceviri_output = gr.Textbox(label="Çeviri", lines=5)
        
        ceviri_button.click(
            ceviri_demo, 
            inputs=[ceviri_input, kaynak_dil, hedef_dil], 
            outputs=ceviri_output
        )
    
    with gr.Tab("Varlık İsmi Tanıma (NER)"):
        with gr.Row():
            with gr.Column():
                ner_input = gr.Textbox(
                    label="Metin", 
                    placeholder="Metni buraya yazın...",
                    value="Apple Inc. is planning to open a new store in Istanbul, Turkey next year. CEO Tim Cook announced this during his visit to Berlin, Germany."
                )
                ner_button = gr.Button("Varlıkları Tanı")
            
            with gr.Column():
                ner_output = gr.Textbox(label="Bulunan Varlıklar", lines=10)
        
        ner_button.click(ner_demo, inputs=ner_input, outputs=ner_output)
    
    # GPU bilgisi göster
    if torch.cuda.is_available():
        gpu_info = f"🚀 GPU: {torch.cuda.get_device_name(0)}"
    else:
        gpu_info = "⚠️ GPU bulunamadı. İşlemler CPU üzerinde çalışacak."
    
    gr.Markdown(f"### Sistem Bilgisi\n{gpu_info}")

# Demo'yu çalıştır
if __name__ == "__main__":
    demo.launch(share=True)  # share=True Gradio'nun geçici bir public URL oluşturmasını sağlar 