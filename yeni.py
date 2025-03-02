import gradio as gr
import torch
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Cihaz kontrolü
device = 0 if torch.cuda.is_available() else -1
print(f"Cihaz: {'CUDA' if device == 0 else 'CPU'}")

# Pipeline önbelleği
class PipelineCache:
    def __init__(self):
        self.pipelines = {}

pipeline_cache = PipelineCache()

# Duygu analizi demo fonksiyonu
def duygu_analizi_demo(text, lang):
    """Duygu analizi pipeline demo fonksiyonu"""
    if not text.strip():
        return """
        <div class="alert alert-warning">
            <div class="alert-icon">⚠️</div>
            <div class="alert-content">
                <div class="alert-title">Uyarı</div>
                <div class="alert-message">Lütfen analiz edilecek bir metin girin.</div>
            </div>
        </div>
        """
    
    try:
        # Dile göre model seç
        if lang == "tr":
            model_name = "savasy/bert-base-turkish-sentiment-cased"
        else:  # Varsayılan İngilizce
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        # Önbellekten pipeline al
        pipeline_key = f"sentiment-analysis-{lang}"
        if pipeline_key not in pipeline_cache.pipelines:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            pipeline_cache.pipelines[pipeline_key] = pipeline(
                "sentiment-analysis", 
                model=model_name,
                device=device
            )
        
        duygu_analizi = pipeline_cache.pipelines[pipeline_key]
        
        # Duygu analizi yap
        sonuc = duygu_analizi(text)
        
        # Sonuçları formatla
        label = sonuc[0]["label"]
        score = sonuc[0]["score"]
        
        # Türkçe model için etiketleri çevir
        if lang == "tr":
            sentiment_class = {
                "LABEL_0": "Olumsuz",
                "LABEL_1": "Tarafsız",
                "LABEL_2": "Olumlu"
            }.get(label, label)
            
            sentiment_color = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive"
            }.get(label, "neutral")
        else:
            sentiment_class = {
                "NEGATIVE": "Olumsuz",
                "POSITIVE": "Olumlu"
            }.get(label, label)
            
            sentiment_color = {
                "NEGATIVE": "negative",
                "POSITIVE": "positive"
            }.get(label, "neutral")
        
        # Emoji seçimi
        emoji = {
            "positive": "😊",
            "neutral": "😐",
            "negative": "😔"
        }.get(sentiment_color, "😐")
        
        # Renk seçimi
        color = {
            "positive": "#0891b2",  # Koyu Turkuaz
            "neutral": "#475569",   # Koyu Gri
            "negative": "#b91c1c"   # Koyu Kırmızı
        }.get(sentiment_color, "#475569")
        
        # Arka plan rengi
        bg_color = {
            "positive": "#ecfeff",  # Açık Turkuaz
            "neutral": "#f1f5f9",   # Açık Gri
            "negative": "#fee2e2"   # Açık Kırmızı
        }.get(sentiment_color, "#f1f5f9")
        
        # HTML çıktısı
        html_output = f"""
        <div class="result-container">
            <div class="result-card">
                <div class="result-header" style="background-color: {color};">
                    <div class="result-emoji">{emoji}</div>
                    <div class="result-title">Duygu Analizi Sonucu</div>
                </div>
                
                <div class="result-body">
                    <div class="result-text-container" style="border-left-color: {color}; background-color: {bg_color};">
                        <div class="result-text">{text}</div>
                    </div>
                    
                    <div class="result-summary">
                        <div class="result-sentiment" style="color: {color};">{sentiment_class}</div>
                        <div class="result-score">Güven skoru: {score:.2f}</div>
                    </div>
                    
                    <div class="result-meter-container">
                        <div class="result-meter">
                            <div class="result-meter-fill" style="width: {score*100}%; background-color: {color};"></div>
                        </div>
                        <div class="result-meter-labels">
                            <div>0</div>
                            <div>0.5</div>
                            <div>1</div>
                        </div>
                    </div>
                    
                    <div class="result-footer">
                        <div class="result-model">
                            <span>Model:</span>
                            <span>{model_name}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html_output
    
    except Exception as e:
        import traceback
        return f"""
        <div class="alert alert-error">
            <div class="alert-icon">❌</div>
            <div class="alert-content">
                <div class="alert-title">Hata</div>
                <div class="alert-message">{str(e)}</div>
                <div class="error-details">
                    <details>
                        <summary>Hata detayları</summary>
                        <pre>{traceback.format_exc()}</pre>
                    </details>
                </div>
            </div>
        </div>
        """

# Çeviri demo fonksiyonu
def ceviri_demo(text, source_lang, target_lang):
    """Çeviri pipeline demo fonksiyonu"""
    if not text.strip():
        return """
        <div class="alert alert-warning">
            <div class="alert-icon">⚠️</div>
            <div class="alert-content">
                <div class="alert-title">Uyarı</div>
                <div class="alert-message">Lütfen çevrilecek bir metin girin.</div>
            </div>
        </div>
        """
    
    try:
        # Model ve tokenizer yükleme
        model_name = "facebook/m2m100_418M"
        
        # Önbellekten model ve tokenizer al
        pipeline_key = f"translation-{source_lang}-{target_lang}"
        if pipeline_key not in pipeline_cache.pipelines:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device if device == 0 else "cpu")
            
            pipeline_cache.pipelines[pipeline_key] = (model, tokenizer)
        else:
            model, tokenizer = pipeline_cache.pipelines[pipeline_key]
        
        # Çeviri yap
        tokenizer.src_lang = source_lang
        encoded = tokenizer(text, return_tensors="pt").to(device if device == 0 else "cpu")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            max_length=128
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Dil isimlerini al
        dil_isimleri = {
            "tr": "Türkçe",
            "en": "İngilizce",
            "de": "Almanca",
            "fr": "Fransızca",
            "es": "İspanyolca",
            "it": "İtalyanca",
            "ru": "Rusça",
            "zh": "Çince",
            "ja": "Japonca",
            "ar": "Arapça",
            "pt": "Portekizce",
            "ko": "Korece"
        }
        
        source_lang_name = dil_isimleri.get(source_lang, source_lang)
        target_lang_name = dil_isimleri.get(target_lang, target_lang)
        
        # HTML çıktısı
        html_output = f"""
        <div class="result-container">
            <div class="result-card">
                <div class="result-header" style="background-color: #0891b2;">
                    <div class="result-emoji">🌐</div>
                    <div class="result-title">Çeviri Sonucu</div>
                </div>
                
                <div class="result-body">
                    <div class="result-text-container" style="border-left-color: #0891b2; background-color: #f0f9ff;">
                        <div class="result-text">{text}</div>
                    </div>
                    
                    <div style="display: flex; justify-content: center; margin: var(--spacing-4) 0;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-weight: 500; color: var(--gray-700);">{source_lang_name}</span>
                            <span style="margin: 0 var(--spacing-2);">→</span>
                            <span style="font-weight: 500; color: var(--primary);">{target_lang_name}</span>
                        </div>
                    </div>
                    
                    <div class="result-text-container" style="border-left-color: #0891b2; background-color: #ecfeff;">
                        <div class="result-text">{translated_text}</div>
                    </div>
                    
                    <div class="result-footer">
                        <div class="result-model">
                            <span>Model:</span>
                            <span>{model_name}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html_output
    
    except Exception as e:
        import traceback
        return f"""
        <div class="alert alert-error">
            <div class="alert-icon">❌</div>
            <div class="alert-content">
                <div class="alert-title">Hata</div>
                <div class="alert-message">{str(e)}</div>
                <details class="error-details">
                    <summary>Detaylar</summary>
                    <pre>{traceback.format_exc()}</pre>
                </details>
            </div>
        </div>
        """

# Metin sınıflandırma demo fonksiyonu
def metin_siniflandirma_demo(text, lang):
    """Metin sınıflandırma pipeline demo fonksiyonu"""
    if not text.strip():
        return """
        <div class="alert alert-warning">
            <div class="alert-icon">⚠️</div>
            <div class="alert-content">
                <div class="alert-title">Uyarı</div>
                <div class="alert-message">Lütfen sınıflandırılacak bir metin girin.</div>
            </div>
        </div>
        """
    
    try:
        # Dile göre model seç
        if lang == "tr":
            model_name = "savasy/bert-base-turkish-text-classification"
            labels = ["ekonomi", "magazin", "saglik", "siyasi", "spor", "teknoloji"]
        else:  # Varsayılan İngilizce
            model_name = "facebook/bart-large-mnli"
            labels = ["business", "entertainment", "health", "politics", "sports", "technology"]
        
        # Önbellekten pipeline al
        pipeline_key = f"text-classification-{lang}"
        if pipeline_key not in pipeline_cache.pipelines:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            if lang == "tr":
                pipeline_cache.pipelines[pipeline_key] = pipeline(
                    "text-classification", 
                    model=model_name,
                    device=device
                )
            else:
                # Zero-shot sınıflandırma için
                pipeline_cache.pipelines[pipeline_key] = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=device
                )
        
        siniflandirma = pipeline_cache.pipelines[pipeline_key]
        
        # Sınıflandırma yap
        if lang == "tr":
            sonuc = siniflandirma(text)
            label = sonuc[0]["label"]
            score = sonuc[0]["score"]
            
            # Kategori renkleri
            category_colors = {
                "ekonomi": "#0891b2",    # Turkuaz
                "magazin": "#8b5cf6",    # Mor
                "saglik": "#10b981",     # Yeşil
                "siyasi": "#ef4444",     # Kırmızı
                "spor": "#f59e0b",       # Turuncu
                "teknoloji": "#3b82f6"   # Mavi
            }
            
            color = category_colors.get(label, "#475569")
            
            # Kategori emojileri
            category_emojis = {
                "ekonomi": "💰",
                "magazin": "🎭",
                "saglik": "🏥",
                "siyasi": "🏛️",
                "spor": "⚽",
                "teknoloji": "💻"
            }
            
            emoji = category_emojis.get(label, "📄")
            
            # Türkçe kategori isimleri
            category_names = {
                "ekonomi": "Ekonomi",
                "magazin": "Magazin",
                "saglik": "Sağlık",
                "siyasi": "Siyaset",
                "spor": "Spor",
                "teknoloji": "Teknoloji"
            }
            
            category_name = category_names.get(label, label)
            
            # HTML çıktısı
            html_output = f"""
            <div class="result-container">
                <div class="result-card">
                    <div class="result-header" style="background-color: {color};">
                        <div class="result-emoji">{emoji}</div>
                        <div class="result-title">Metin Sınıflandırma Sonucu</div>
                    </div>
                    
                    <div class="result-body">
                        <div class="result-text-container" style="border-left-color: {color}; background-color: #f8fafc;">
                            <div class="result-text">{text}</div>
                        </div>
                        
                        <div class="result-summary">
                            <div class="result-category" style="color: {color};">{category_name}</div>
                            <div class="result-score">Güven skoru: {score:.2f}</div>
                        </div>
                        
                        <div class="result-meter-container">
                            <div class="result-meter">
                                <div class="result-meter-fill" style="width: {score*100}%; background-color: {color};"></div>
                            </div>
                            <div class="result-meter-labels">
                                <div>0</div>
                                <div>0.5</div>
                                <div>1</div>
                            </div>
                        </div>
                        
                        <div class="result-footer">
                            <div class="result-model">
                                <span>Model:</span>
                                <span>{model_name}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
        else:
            # Zero-shot sınıflandırma
            sonuc = siniflandirma(text, labels)
            
            # En yüksek skorlu kategoriyi bul
            top_label = sonuc["labels"][0]
            top_score = sonuc["scores"][0]
            
            # Kategori renkleri
            category_colors = {
                "business": "#0891b2",      # Turkuaz
                "entertainment": "#8b5cf6",  # Mor
                "health": "#10b981",         # Yeşil
                "politics": "#ef4444",       # Kırmızı
                "sports": "#f59e0b",         # Turuncu
                "technology": "#3b82f6"      # Mavi
            }
            
            color = category_colors.get(top_label, "#475569")
            
            # Kategori emojileri
            category_emojis = {
                "business": "💰",
                "entertainment": "🎭",
                "health": "🏥",
                "politics": "🏛️",
                "sports": "⚽",
                "technology": "💻"
            }
            
            emoji = category_emojis.get(top_label, "📄")
            
            # İngilizce kategori isimleri Türkçe karşılıkları
            category_names = {
                "business": "İş Dünyası",
                "entertainment": "Eğlence",
                "health": "Sağlık",
                "politics": "Siyaset",
                "sports": "Spor",
                "technology": "Teknoloji"
            }
            
            category_name = category_names.get(top_label, top_label)
            
            # Tüm kategorilerin sonuçlarını hazırla
            all_categories = ""
            for i in range(len(sonuc["labels"])):
                label = sonuc["labels"][i]
                score = sonuc["scores"][i]
                cat_color = category_colors.get(label, "#475569")
                cat_name = category_names.get(label, label)
                
                all_categories += f"""
                <div style="margin-bottom: var(--spacing-2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--spacing-1);">
                        <span style="font-weight: 500; color: {cat_color};">{cat_name}</span>
                        <span style="font-size: 0.875rem; color: var(--gray-500);">{score:.2f}</span>
                    </div>
                    <div class="result-meter">
                        <div class="result-meter-fill" style="width: {score*100}%; background-color: {cat_color};"></div>
                    </div>
                </div>
                """
            
            # HTML çıktısı
            html_output = f"""
            <div class="result-container">
                <div class="result-card">
                    <div class="result-header" style="background-color: {color};">
                        <div class="result-emoji">{emoji}</div>
                        <div class="result-title">Metin Sınıflandırma Sonucu</div>
                    </div>
                    
                    <div class="result-body">
                        <div class="result-text-container" style="border-left-color: {color}; background-color: #f8fafc;">
                            <div class="result-text">{text}</div>
                        </div>
                        
                        <div class="result-summary">
                            <div class="result-category" style="color: {color};">{category_name}</div>
                            <div class="result-score">Güven skoru: {top_score:.2f}</div>
                        </div>
                        
                        <h4 style="margin-top: var(--spacing-6); margin-bottom: var(--spacing-3); font-weight: 600; color: var(--gray-700);">Tüm Kategoriler</h4>
                        
                        {all_categories}
                        
                        <div class="result-footer">
                            <div class="result-model">
                                <span>Model:</span>
                                <span>{model_name}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        return html_output
    
    except Exception as e:
        import traceback
        return f"""
        <div class="alert alert-error">
            <div class="alert-icon">❌</div>
            <div class="alert-content">
                <div class="alert-title">Hata</div>
                <div class="alert-message">{str(e)}</div>
                <details class="error-details">
                    <summary>Detaylar</summary>
                    <pre>{traceback.format_exc()}</pre>
                </details>
            </div>
        </div>
        """

# CSS stilleri
custom_css = """
:root {
    --primary: #3b82f6;
    --primary-dark: #2563eb;
    --primary-light: #dbeafe;
    --secondary: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --warning-light: #fef3c7;
    --error: #ef4444;
    --error-light: #fee2e2;
    
    --gray-50: #f8fafc;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-300: #cbd5e1;
    --gray-400: #94a3b8;
    --gray-500: #64748b;
    --gray-600: #475569;
    --gray-700: #334155;
    --gray-800: #1e293b;
    --gray-900: #0f172a;
    
    --radius-sm: 0.25rem;
    --radius: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    
    --spacing-1: 0.25rem;
    --spacing-2: 0.5rem;
    --spacing-3: 0.75rem;
    --spacing-4: 1rem;
    --spacing-5: 1.25rem;
    --spacing-6: 1.5rem;
    --spacing-8: 2rem;
    --spacing-10: 2.5rem;
    --spacing-12: 3rem;
    
    --transition: all 0.2s ease-in-out;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    color: var(--gray-800);
    background-color: var(--gray-50);
    line-height: 1.5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-4);
}

/* Kategori Kartları */
.categories {
    display: flex;
    justify-content: center;
    gap: var(--spacing-6);
    margin-bottom: var(--spacing-8);
    flex-wrap: wrap;
}

.category-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing-4);
    cursor: pointer;
    transition: var(--transition);
    border-radius: var(--radius);
    width: 100px;
}

.category-card:hover {
    background-color: var(--gray-100);
}

.category-card.active {
    background-color: var(--primary-light);
}

.category-icon {
    font-size: 2rem;
    margin-bottom: var(--spacing-2);
    color: var(--primary);
}

.category-name {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--gray-700);
    text-align: center;
}

/* Sekme Navigasyonu */
.tab-nav {
    display: flex;
    border-bottom: 1px solid var(--gray-200);
    margin-bottom: var(--spacing-6);
    overflow-x: auto;
    scrollbar-width: none;
}

.tab-nav::-webkit-scrollbar {
    display: none;
}

.tab-item {
    padding: var(--spacing-4) var(--spacing-6);
    font-weight: 500;
    color: var(--gray-600);
    border-bottom: 2px solid transparent;
    cursor: pointer;
    white-space: nowrap;
    transition: var(--transition);
}

.tab-item:hover {
    color: var(--primary);
}

.tab-item.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
}

/* Form Elemanları */
.form-group {
    margin-bottom: var(--spacing-4);
}

.form-label {
    display: block;
    margin-bottom: var(--spacing-2);
    font-weight: 500;
    color: var(--gray-700);
}

.form-control {
    width: 100%;
    padding: var(--spacing-3) var(--spacing-4);
    border: 1px solid var(--gray-300);
    border-radius: var(--radius);
    background-color: white;
    color: var(--gray-800);
    font-size: 1rem;
    transition: var(--transition);
}

.form-control:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--primary-light);
}

.form-control::placeholder {
    color: var(--gray-400);
}

.form-select {
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2364748b'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.75rem center;
    background-size: 1rem;
    padding-right: 2.5rem;
}

/* Butonlar */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-2) var(--spacing-4);
    border: none;
    border-radius: var(--radius);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition);
}

.btn-primary {
    background-color: var(--primary);
    color: white;
}

.btn-primary:hover {
    background-color: var(--primary-dark);
}

.btn-lg {
    padding: var(--spacing-3) var(--spacing-6);
    font-size: 1rem;
}

/* Sonuç Kartları */
.result-container {
    margin-top: var(--spacing-6);
}

.result-card {
    background-color: white;
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow);
}

.result-header {
    display: flex;
    align-items: center;
    padding: var(--spacing-4) var(--spacing-6);
    color: white;
    font-weight: 600;
}

.result-emoji {
    font-size: 1.5rem;
    margin-right: var(--spacing-3);
}

.result-title {
    font-size: 1.125rem;
}

.result-body {
    padding: var(--spacing-6);
}

.result-text-container {
    padding: var(--spacing-4);
    border-radius: var(--radius);
    border-left: 4px solid;
    margin-bottom: var(--spacing-4);
}

.result-text {
    font-size: 1rem;
    line-height: 1.6;
}

.result-summary {
    display: flex;
    align-items: baseline;
    margin-bottom: var(--spacing-4);
}

.result-sentiment, .result-category {
    font-size: 1.25rem;
    font-weight: 600;
    margin-right: var(--spacing-3);
}

.result-score {
    font-size: 0.875rem;
    color: var(--gray-500);
}

.result-meter-container {
    margin-bottom: var(--spacing-6);
}

.result-meter {
    height: 8px;
    background-color: var(--gray-200);
    border-radius: 9999px;
    overflow: hidden;
    margin-bottom: var(--spacing-2);
}

.result-meter-fill {
    height: 100%;
    border-radius: 9999px;
}

.result-meter-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--gray-500);
}

.result-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: var(--spacing-4);
    border-top: 1px solid var(--gray-200);
    font-size: 0.875rem;
    color: var(--gray-500);
}

.result-model {
    display: flex;
    align-items: center;
    gap: var(--spacing-2);
}

/* Uyarı Kutuları */
.alert {
    display: flex;
    padding: var(--spacing-4);
    border-radius: var(--radius);
    margin-bottom: var(--spacing-4);
}

.alert-warning {
    background-color: var(--warning-light);
    color: var(--warning);
}

.alert-error {
    background-color: var(--error-light);
    color: var(--error);
}

.alert-icon {
    font-size: 1.25rem;
    margin-right: var(--spacing-3);
}

.alert-content {
    flex: 1;
}

.alert-title {
    font-weight: 600;
    margin-bottom: var(--spacing-1);
}

.alert-message {
    font-size: 0.875rem;
}

.error-details {
    margin-top: var(--spacing-2);
    font-size: 0.75rem;
}

.error-details summary {
    cursor: pointer;
}

/* Kategori Navigasyonu */
.category-nav {
    display: flex;
    justify-content: center;
    gap: var(--spacing-6);
    margin-bottom: var(--spacing-8);
    flex-wrap: wrap;
}

.category-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    transition: var(--transition);
    padding: var(--spacing-2);
    border-radius: var(--radius);
    width: 100px;
}

.category-item:hover {
    background-color: var(--gray-100);
}

.category-item.active {
    background-color: var(--primary-light);
}

.category-icon {
    font-size: 2rem;
    margin-bottom: var(--spacing-2);
    color: var(--primary);
}

.category-name {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--gray-700);
    text-align: center;
}

/* Tab Navigasyonu */
.tab-nav {
    display: flex;
    border-bottom: 1px solid var(--gray-200);
    margin-bottom: var(--spacing-6);
    overflow-x: auto;
    scrollbar-width: none;
}

.tab-nav::-webkit-scrollbar {
    display: none;
}

.tab-item {
    padding: var(--spacing-4) var(--spacing-6);
    font-weight: 500;
    color: var(--gray-600);
    border-bottom: 2px solid transparent;
    cursor: pointer;
    white-space: nowrap;
    transition: var(--transition);
}

.tab-item:hover {
    color: var(--primary);
}

.tab-item.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
}

/* Responsive */
@media (max-width: 768px) {
    .container {
        padding: 0 var(--spacing-2);
    }
    
    .header {
        padding: var(--spacing-6) var(--spacing-2);
    }
    
    .title {
        font-size: 1.5rem;
    }
    
    .subtitle {
        font-size: 0.875rem;
    }
    
    .category-nav {
        gap: var(--spacing-2);
    }
    
    .category-item {
        width: 80px;
    }
    
    .category-icon {
        font-size: 1.5rem;
    }
    
    .category-name {
        font-size: 0.75rem;
    }
    
    .tab-item {
        padding: var(--spacing-3) var(--spacing-4);
    }
}
"""

# Gradio arayüzü
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    # Aktif sekme state'i
    active_tab = gr.State("duygu-analizi")
    
    with gr.Column(elem_classes="container"):
        # Başlık
        gr.HTML("""
        <div class="header">
            <h1 class="title">NLP Asistanı</h1>
            <p class="subtitle">Doğal dil işleme görevleri için yapay zeka destekli araçlar</p>
        </div>
        """)
        
        # Kategori Navigasyonu
        gr.HTML("""
        <div class="category-nav">
            <div class="category-item active" data-tab="duygu-analizi">
                <div class="category-icon">😊</div>
                <div class="category-name">Duygu Analizi</div>
            </div>
            <div class="category-item" data-tab="ceviri">
                <div class="category-icon">🌐</div>
                <div class="category-name">Çeviri</div>
            </div>
            <div class="category-item" data-tab="siniflandirma">
                <div class="category-icon">📊</div>
                <div class="category-name">Sınıflandırma</div>
            </div>
            <div class="category-item" data-tab="ozet">
                <div class="category-icon">📝</div>
                <div class="category-name">Özet Çıkarma</div>
            </div>
            <div class="category-item" data-tab="soru-cevap">
                <div class="category-icon">💬</div>
                <div class="category-name">Soru Cevaplama</div>
            </div>
            <div class="category-item" data-tab="varlik-tanima">
                <div class="category-icon">🏷️</div>
                <div class="category-name">Varlık Tanıma</div>
            </div>
        </div>
        """)
        
        # Tab Navigasyonu
        gr.HTML("""
        <div class="tab-nav">
            <div class="tab-item active" data-tab="duygu-analizi">Duygu Analizi</div>
            <div class="tab-item" data-tab="ceviri">Çeviri</div>
            <div class="tab-item" data-tab="siniflandirma">Metin Sınıflandırma</div>
        </div>
        """)
        
        # Duygu Analizi Tab İçeriği
        with gr.Column(elem_classes="tab-content active", elem_id="duygu-analizi-tab"):
            duygu_text = gr.Textbox(
                label="Analiz edilecek metin",
                placeholder="Analiz etmek istediğiniz metni buraya yazın...",
                lines=5,
                elem_classes="form-control"
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    duygu_lang = gr.Dropdown(
                        choices=[("Türkçe", "tr"), ("İngilizce", "en")],
                        value="tr",
                        label="Dil",
                        elem_classes="form-control form-select"
                    )
                
                with gr.Column(scale=1):
                    duygu_button = gr.Button("Analiz Et", elem_classes="btn btn-primary btn-lg")
            
            duygu_output = gr.HTML(elem_classes="result-output")
        
        # Çeviri Tab İçeriği
        with gr.Column(elem_classes="tab-content", elem_id="ceviri-tab", visible=False):
            ceviri_text = gr.Textbox(
                label="Çevrilecek metin",
                placeholder="Çevirmek istediğiniz metni buraya yazın...",
                lines=5,
                elem_classes="form-control"
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    ceviri_source_lang = gr.Dropdown(
                        choices=[
                            ("Türkçe", "tr"), 
                            ("İngilizce", "en"),
                            ("Almanca", "de"),
                            ("Fransızca", "fr"),
                            ("İspanyolca", "es")
                        ],
                        value="tr",
                        label="Kaynak Dil",
                        elem_classes="form-control form-select"
                    )
                
                with gr.Column(scale=1):
                    gr.HTML('<div style="display: flex; align-items: center; justify-content: center; height: 100%;">→</div>')
                
                with gr.Column(scale=2):
                    ceviri_target_lang = gr.Dropdown(
                        choices=[
                            ("Türkçe", "tr"), 
                            ("İngilizce", "en"),
                            ("Almanca", "de"),
                            ("Fransızca", "fr"),
                            ("İspanyolca", "es")
                        ],
                        value="en",
                        label="Hedef Dil",
                        elem_classes="form-control form-select"
                    )
            
            ceviri_button = gr.Button("Çevir", elem_classes="btn btn-primary btn-lg")
            
            ceviri_output = gr.HTML(elem_classes="result-output")
        
        # Sınıflandırma Tab İçeriği
        with gr.Column(elem_classes="tab-content", elem_id="siniflandirma-tab", visible=False):
            sinif_text = gr.Textbox(
                label="Sınıflandırılacak metin",
                placeholder="Sınıflandırmak istediğiniz metni buraya yazın...",
                lines=5,
                elem_classes="form-control"
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    sinif_lang = gr.Dropdown(
                        choices=[("Türkçe", "tr"), ("İngilizce", "en")],
                        value="tr",
                        label="Dil",
                        elem_classes="form-control form-select"
                    )
                
                with gr.Column(scale=1):
                    sinif_button = gr.Button("Sınıflandır", elem_classes="btn btn-primary btn-lg")
            
            sinif_output = gr.HTML(elem_classes="result-output")
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>© 2024 NLP Asistanı | Hugging Face Transformers ve Gradio ile geliştirilmiştir.</p>
        </div>
        """)
    
    # JavaScript ile sekme geçişleri
    gr.HTML("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Kategori kartları için tıklama olayları
        const categoryItems = document.querySelectorAll('.category-item');
        categoryItems.forEach(item => {
            item.addEventListener('click', function() {
                const tab = this.getAttribute('data-tab');
                
                // Aktif kategori kartını güncelle
                document.querySelectorAll('.category-item').forEach(el => {
                    el.classList.remove('active');
                });
                this.classList.add('active');
                
                // Aktif sekmeyi güncelle
                document.querySelectorAll('.tab-item').forEach(el => {
                    el.classList.remove('active');
                    if (el.getAttribute('data-tab') === tab) {
                        el.classList.add('active');
                    }
                });
                
                // Sekme içeriğini güncelle
                document.querySelectorAll('.tab-content').forEach(el => {
                    el.style.display = 'none';
                });
                document.getElementById(tab + '-tab').style.display = 'block';
            });
        });
        
        // Sekme navigasyonu için tıklama olayları
        const tabItems = document.querySelectorAll('.tab-item');
        tabItems.forEach(item => {
            item.addEventListener('click', function() {
                const tab = this.getAttribute('data-tab');
                
                // Aktif sekmeyi güncelle
                document.querySelectorAll('.tab-item').forEach(el => {
                    el.classList.remove('active');
                });
                this.classList.add('active');
                
                // Aktif kategori kartını güncelle
                document.querySelectorAll('.category-item').forEach(el => {
                    el.classList.remove('active');
                    if (el.getAttribute('data-tab') === tab) {
                        el.classList.add('active');
                    }
                });
                
                // Sekme içeriğini güncelle
                document.querySelectorAll('.tab-content').forEach(el => {
                    el.style.display = 'none';
                });
                document.getElementById(tab + '-tab').style.display = 'block';
            });
        });
    });
    </script>
    """)
    
    # Buton Tıklama Olayları
    duygu_button.click(
        fn=duygu_analizi_demo,
        inputs=[duygu_text, duygu_lang],
        outputs=duygu_output
    )
    
    ceviri_button.click(
        fn=ceviri_demo,
        inputs=[ceviri_text, ceviri_source_lang, ceviri_target_lang],
        outputs=ceviri_output
    )
    
    sinif_button.click(
        fn=metin_siniflandirma_demo,
        inputs=[sinif_text, sinif_lang],
        outputs=sinif_output
    )

# Uygulamayı başlat
if __name__ == "__main__":
    demo.launch(share=True)