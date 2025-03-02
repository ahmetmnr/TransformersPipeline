document.addEventListener('DOMContentLoaded', function() {
    // Tab değiştirme işlevi
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Aktif tab'ı değiştir
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // İlgili içeriği göster
            const tabId = button.getAttribute('data-tab');
            tabPanes.forEach(pane => {
                pane.classList.remove('active');
                if (pane.id === tabId) {
                    pane.classList.add('active');
                }
            });
        });
    });
    
    // Range slider değerlerini güncelleme
    const rangeSliders = document.querySelectorAll('input[type="range"]');
    rangeSliders.forEach(slider => {
        const valueDisplay = document.getElementById(`${slider.id}-value`);
        if (valueDisplay) {
            valueDisplay.textContent = slider.value;
            slider.addEventListener('input', () => {
                valueDisplay.textContent = slider.value;
            });
        }
    });
    
    // Yükleniyor göstergesi oluşturma fonksiyonu
    function createLoadingIndicator() {
        const loading = document.createElement('div');
        loading.className = 'loading';
        return loading;
    }
    
    // API isteği gönderme yardımcı fonksiyonu
    async function sendApiRequest(endpoint, data, outputElement, processResponse) {
        // Yükleniyor göstergesi
        outputElement.innerHTML = '';
        const loading = createLoadingIndicator();
        outputElement.appendChild(loading);
        
        try {
            const response = await fetch(`/api/${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            // Hata kontrolü
            if (result.error) {
                outputElement.innerHTML = `<div class="result-card">
                    <div class="result-header sentiment-negative">Hata</div>
                    <div class="result-content">${result.error}</div>
                </div>`;
                return;
            }
            
            // Sonucu işle
            processResponse(result, outputElement);
            
        } catch (error) {
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-negative">Hata</div>
                <div class="result-content">İstek sırasında bir hata oluştu: ${error.message}</div>
            </div>`;
        }
    }
    
    // Dosya yükleme yardımcı fonksiyonu
    async function sendFileRequest(endpoint, formData, outputElement, processResponse) {
        // Yükleniyor göstergesi
        outputElement.innerHTML = '';
        const loading = createLoadingIndicator();
        outputElement.appendChild(loading);
        
        try {
            const response = await fetch(`/api/${endpoint}`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            // Hata kontrolü
            if (result.error) {
                outputElement.innerHTML = `<div class="result-card">
                    <div class="result-header sentiment-negative">Hata</div>
                    <div class="result-content">${result.error}</div>
                </div>`;
                return;
            }
            
            // Sonucu işle
            processResponse(result, outputElement);
            
        } catch (error) {
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-negative">Hata</div>
                <div class="result-content">İstek sırasında bir hata oluştu: ${error.message}</div>
            </div>`;
        }
    }
    
    // Duygu Analizi
    const duyguButton = document.getElementById('duygu-button');
    const duyguOutput = document.getElementById('duygu-output');
    
    duyguButton.addEventListener('click', () => {
        const metin = document.getElementById('duygu-metin').value;
        const dil = document.querySelector('input[name="duygu-dil"]:checked').value;
        
        sendApiRequest('duygu-analizi', { metin, dil }, duyguOutput, (result, outputElement) => {
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-${result.sentiment_class}">
                    ${result.label} (Güven: ${result.score.toFixed(4)})
                </div>
                <div class="result-content">
                    <p>"${result.metin}"</p>
                </div>
            </div>`;
        });
    });
    
    // Metin Üretme
    const uretmeButton = document.getElementById('uretme-button');
    const uretilenOutput = document.getElementById('uretilen-output');
    
    uretmeButton.addEventListener('click', () => {
        const baslangicMetni = document.getElementById('baslangic-metin').value;
        const maxUzunluk = parseInt(document.getElementById('max-uzunluk').value);
        const tekrarSayisi = parseInt(document.getElementById('tekrar-sayisi').value);
        
        sendApiRequest('metin-uretme', { 
            baslangic_metni: baslangicMetni, 
            max_uzunluk: maxUzunluk, 
            tekrar_sayisi: tekrarSayisi 
        }, uretilenOutput, (result, outputElement) => {
            let html = '';
            result.sonuclar.forEach((sonuc, index) => {
                html += `<div class="result-card">
                    <div class="result-content">
                        <strong>Sonuç ${index + 1}:</strong>
                        <p>${sonuc}</p>
                    </div>
                </div>`;
            });
            outputElement.innerHTML = html;
        });
    });
    
    // Maskeleme
    const maskButton = document.getElementById('mask-button');
    const maskOutput = document.getElementById('mask-output');
    
    maskButton.addEventListener('click', () => {
        const maskMetin = document.getElementById('mask-metin').value;
        
        sendApiRequest('maskeleme', { maskeli_metin: maskMetin }, maskOutput, (result, outputElement) => {
            let html = '';
            result.sonuclar.forEach((sonuc, index) => {
                html += `<div class="result-card">
                    <div class="result-content">
                        <strong>${index + 1}.</strong> ${sonuc.sequence}<br>
                        <small>Skor: ${sonuc.score.toFixed(4)}</small>
                    </div>
                </div>`;
            });
            outputElement.innerHTML = html;
        });
    });
    
    // Soru Cevaplama
    const soruButton = document.getElementById('soru-button');
    const soruOutput = document.getElementById('soru-output');
    
    soruButton.addEventListener('click', () => {
        const soru = document.getElementById('soru-input').value;
        const metin = document.getElementById('soru-metin').value;
        
        sendApiRequest('soru-cevaplama', { soru, metin }, soruOutput, (result, outputElement) => {
            // Cevap undefined ise veya boş ise
            const cevap = result.cevap || "Metinde bu sorunun cevabı bulunamadı.";
            
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Cevap (Güven: ${result.score.toFixed(4)})
                </div>
                <div class="result-content">
                    <p><strong>Soru:</strong> ${result.soru}</p>
                    <p><strong>Cevap:</strong> ${cevap}</p>
                </div>
            </div>`;
        });
    });
    
    // Özetleme
    const ozetButton = document.getElementById('ozet-button');
    const ozetOutput = document.getElementById('ozet-output');
    
    if (ozetButton) {  // Eleman varsa işlemi gerçekleştir
        ozetButton.addEventListener('click', () => {
            const uzunMetin = document.getElementById('uzun-metin')?.value || '';
            const minUzunluk = parseInt(document.getElementById('min-uzunluk')?.value || '30');
            const maxUzunluk = parseInt(document.getElementById('max-ozet-uzunluk')?.value || '100');
            
            // Metin boş mu kontrol et
            if (!uzunMetin.trim()) {
                ozetOutput.innerHTML = `<div class="result-card">
                    <div class="result-header sentiment-negative">Hata</div>
                    <div class="result-content">Lütfen özetlenecek bir metin girin!</div>
                </div>`;
                return;
            }
            
            // Metin çok kısa mı kontrol et
            if (uzunMetin.split(/\s+/).length < 20) {
                ozetOutput.innerHTML = `<div class="result-card">
                    <div class="result-header sentiment-negative">Hata</div>
                    <div class="result-content">Metin özetleme için çok kısa. Lütfen daha uzun bir metin girin.</div>
                </div>`;
                return;
            }
            
            console.log('Özetleme isteği gönderiliyor:', { metin: uzunMetin, min_uzunluk: minUzunluk, max_uzunluk: maxUzunluk });
            
            sendApiRequest('ozetleme', { 
                metin: uzunMetin, 
                min_uzunluk: minUzunluk, 
                max_uzunluk: maxUzunluk 
            }, ozetOutput, (result, outputElement) => {
                outputElement.innerHTML = `<div class="result-card">
                    <div class="result-header sentiment-neutral">
                        Özet
                    </div>
                    <div class="result-content">
                        <p>${result.ozet}</p>
                        <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                        <p><small>Orijinal Metin Uzunluğu: ${result.original_length} karakter</small></p>
                        <p><small>Özet Uzunluğu: ${result.summary_length} karakter</small></p>
                        <p><small>Sıkıştırma Oranı: %${result.compression_ratio.toFixed(2)}</small></p>
                    </div>
                </div>`;
            });
        });
    }
    
    // Çeviri
    const ceviriButton = document.getElementById('ceviri-button');
    const ceviriOutput = document.getElementById('ceviri-output');
    
    ceviriButton.addEventListener('click', () => {
        const metin = document.getElementById('ceviri-metin')?.value || '';
        const kaynakDil = document.getElementById('kaynak-dil')?.value || 'auto';
        const hedefDil = document.getElementById('hedef-dil')?.value || 'tr';
        
        // Metin boş mu kontrol et
        if (!metin.trim()) {
            ceviriOutput.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-negative">Hata</div>
                <div class="result-content">Lütfen çevrilecek bir metin girin!</div>
            </div>`;
            return;
        }
        
        sendApiRequest('ceviri', { 
            metin: metin, 
            kaynak_dil: kaynakDil, 
            hedef_dil: hedefDil 
        }, ceviriOutput, (result, outputElement) => {
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Çeviri (${result.source_language} → ${result.target_language})
                </div>
                <div class="result-content">
                    <p><strong>Kaynak Metin:</strong> "${result.source_text}"</p>
                    <p><strong>Çeviri:</strong> "${result.translated_text}"</p>
                </div>
            </div>`;
        });
    });
    
    // Varlık İsmi Tanıma
    const nerButton = document.getElementById('ner-button');
    const nerOutput = document.getElementById('ner-output');
    
    nerButton.addEventListener('click', () => {
        const nerMetin = document.getElementById('ner-metin').value;
        
        sendApiRequest('varlik-ismi-tanima', { metin: nerMetin }, nerOutput, (result, outputElement) => {
            // Varlıkları işaretlenmiş metni oluştur
            let markedText = result.text;
            
            // Varlıkları sondan başa doğru işaretle (çakışmaları önlemek için)
            const sortedEntities = [...result.entities].sort((a, b) => b.start - a.start);
            
            sortedEntities.forEach(entity => {
                const entityText = markedText.substring(entity.start, entity.end);
                const markedEntity = `<span class="entity entity-${entity.entity_group.toLowerCase()}">${entityText} <small>(${entity.entity_group})</small></span>`;
                markedText = markedText.substring(0, entity.start) + markedEntity + markedText.substring(entity.end);
            });
            
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Varlık İsmi Tanıma Sonuçları
                </div>
                <div class="result-content">
                    <p>${markedText}</p>
                    <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                    <p><small>Toplam ${result.entities.length} varlık bulundu.</small></p>
                </div>
            </div>`;
        });
    });
    
    // Türkçe Maskeleme
    const turkceMaskButton = document.getElementById('turkce-mask-button');
    const turkceMaskOutput = document.getElementById('turkce-mask-output');
    
    turkceMaskButton.addEventListener('click', () => {
        const maskMetin = document.getElementById('turkce-mask-metin').value;
        
        sendApiRequest('turkce-maskeleme', { maskeli_metin: maskMetin }, turkceMaskOutput, (result, outputElement) => {
            let html = '';
            result.sonuclar.forEach((sonuc, index) => {
                html += `<div class="result-card">
                    <div class="result-content">
                        <strong>${index + 1}.</strong> ${sonuc.sequence}<br>
                        <small>Skor: ${sonuc.score.toFixed(4)}</small>
                    </div>
                </div>`;
            });
            outputElement.innerHTML = html;
        });
    });
    
    // Metin Sınıflandırma
    const siniflandirmaButton = document.getElementById('siniflandirma-button');
    const siniflandirmaOutput = document.getElementById('siniflandirma-output');
    
    siniflandirmaButton.addEventListener('click', () => {
        const metin = document.getElementById('siniflandirma-metin').value;
        const kategoriSayisi = parseInt(document.getElementById('kategori-sayisi').value);
        
        sendApiRequest('metin-siniflandirma', { 
            metin: metin, 
            kategori_sayisi: kategoriSayisi 
        }, siniflandirmaOutput, (result, outputElement) => {
            let html = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Sınıflandırma Sonuçları
                </div>
                <div class="result-content">
                    <p><strong>Metin:</strong> "${result.text}"</p>
                    <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                    <div>`;
            
            result.categories.forEach((category, index) => {
                html += `<div style="margin-bottom: 0.5rem;">
                    <strong>${index + 1}. ${category.label}</strong> (${(category.score * 100).toFixed(2)}%)
                    <div style="height: 10px; background-color: #e5e7eb; border-radius: 5px; overflow: hidden;">
                        <div style="height: 100%; width: ${category.score * 100}%; background-color: var(--primary-color);"></div>
                    </div>
                </div>`;
            });
            
            html += `</div></div></div>`;
            outputElement.innerHTML = html;
        });
    });
    
    // Sıfır-Atış Sınıflandırma
    const sifirAtisButton = document.getElementById('sifir-atis-button');
    const sifirAtisOutput = document.getElementById('sifir-atis-output');
    
    sifirAtisButton.addEventListener('click', () => {
        const metin = document.getElementById('sifir-atis-metin').value;
        const etiketler = document.getElementById('etiketler').value;
        
        sendApiRequest('sifir-atis-siniflandirma', { 
            metin: metin, 
            etiketler: etiketler 
        }, sifirAtisOutput, (result, outputElement) => {
            let html = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Sıfır-Atış Sınıflandırma Sonuçları
                </div>
                <div class="result-content">
                    <p><strong>Metin:</strong> "${result.text}"</p>
                    <p><strong>Etiketler:</strong> ${result.labels.join(', ')}</p>
                    <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                    <div>`;
            
            result.results.forEach((category, index) => {
                html += `<div style="margin-bottom: 0.5rem;">
                    <strong>${index + 1}. ${category.label}</strong> (${(category.score * 100).toFixed(2)}%)
                    <div style="height: 10px; background-color: #e5e7eb; border-radius: 5px; overflow: hidden;">
                        <div style="height: 100%; width: ${category.score * 100}%; background-color: var(--primary-color);"></div>
                    </div>
                </div>`;
            });
            
            html += `</div></div></div>`;
            outputElement.innerHTML = html;
        });
    });
    
    // Cümle Benzerliği
    const benzerlikButton = document.getElementById('benzerlik-button');
    const benzerlikOutput = document.getElementById('benzerlik-output');
    
    benzerlikButton.addEventListener('click', () => {
        const cumle1 = document.getElementById('cumle1').value;
        const cumle2 = document.getElementById('cumle2').value;
        
        sendApiRequest('cumle-benzerligi', { 
            cumle1: cumle1, 
            cumle2: cumle2 
        }, benzerlikOutput, (result, outputElement) => {
            // Benzerlik seviyesine göre renk
            let colorClass = 'sentiment-negative';
            if (result.similarity_percentage >= 75) {
                colorClass = 'sentiment-positive';
            } else if (result.similarity_percentage >= 50) {
                colorClass = 'sentiment-neutral';
            }
            
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header ${colorClass}">
                    Benzerlik: %${result.similarity_percentage.toFixed(2)} (${result.similarity_level})
                </div>
                <div class="result-content">
                    <p><strong>Cümle 1:</strong> "${result.cumle1}"</p>
                    <p><strong>Cümle 2:</strong> "${result.cumle2}"</p>
                    <p><small>Kosinüs Benzerliği: ${result.cosine_similarity.toFixed(4)}</small></p>
                </div>
            </div>`;
        });
    });
    
    // Görsel Soru Cevaplama
    const gorselButton = document.getElementById('gorsel-button');
    const gorselOutput = document.getElementById('gorsel-output');
    const gorselInput = document.getElementById('gorsel-input');
    const gorselPreview = document.getElementById('gorsel-preview');
    
    // Görsel önizleme
    gorselInput.addEventListener('change', () => {
        if (gorselInput.files && gorselInput.files[0]) {
            const reader = new FileReader();
            reader.onload = (e) => {
                gorselPreview.innerHTML = `<img src="${e.target.result}" alt="Seçilen görsel">`;
            };
            reader.readAsDataURL(gorselInput.files[0]);
        }
    });
    
    gorselButton.addEventListener('click', () => {
        if (!gorselInput.files || !gorselInput.files[0]) {
            gorselOutput.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-negative">Hata</div>
                <div class="result-content">Lütfen bir görsel yükleyin!</div>
            </div>`;
            return;
        }
        
        const soru = document.getElementById('gorsel-soru').value;
        const formData = new FormData();
        formData.append('image', gorselInput.files[0]);
        formData.append('soru', soru);
        
        sendFileRequest('gorsel-soru-cevaplama', formData, gorselOutput, (result, outputElement) => {
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Görsel Soru Cevaplama
                </div>
                <div class="result-content">
                    <p><strong>Soru:</strong> ${result.soru}</p>
                    <p><strong>Cevap:</strong> ${result.cevap}</p>
                </div>
            </div>`;
        });
    });
    
    // Metin Düzeltme
    const duzeltmeButton = document.getElementById('duzeltme-button');
    const duzeltmeOutput = document.getElementById('duzeltme-output');
    
    duzeltmeButton.addEventListener('click', () => {
        const metin = document.getElementById('duzeltme-metin').value;
        
        sendApiRequest('metin-duzeltme', { metin: metin }, duzeltmeOutput, (result, outputElement) => {
            // Değişiklikleri göster
            if (result.changes && result.changes.length > 0) {
                let changesHtml = '<div style="margin-top: 1rem;"><strong>Değişiklikler:</strong><ul>';
                
                result.changes.forEach(change => {
                    changesHtml += `<li><span style="color: #ef4444;">${change.original}</span> → <span style="color: #22c55e;">${change.corrected}</span></li>`;
                });
                
                changesHtml += '</ul></div>';
                
                outputElement.innerHTML = `<div class="result-card">
                    <div class="result-header sentiment-neutral">
                        ${result.message}
                    </div>
                    <div class="result-content">
                        <p><strong>Orijinal Metin:</strong> "${result.original_text}"</p>
                        <p><strong>Düzeltilmiş Metin:</strong> "${result.corrected_text}"</p>
                        ${changesHtml}
                    </div>
                </div>`;
            } else {
                // Değişiklik yoksa
                outputElement.innerHTML = `<div class="result-card">
                    <div class="result-header sentiment-positive">
                        ${result.message}
                    </div>
                    <div class="result-content">
                        <p>"${result.original_text}"</p>
                    </div>
                </div>`;
            }
        });
    });
    
    // Çok Sınıflı Duygu Analizi
    const cokDuyguButton = document.getElementById('cok-duygu-button');
    const cokDuyguOutput = document.getElementById('cok-duygu-output');
    
    cokDuyguButton.addEventListener('click', () => {
        const metin = document.getElementById('cok-duygu-metin').value;
        
        sendApiRequest('cok-sinifli-duygu-analizi', { metin: metin }, cokDuyguOutput, (result, outputElement) => {
            // Duygu çubuklarını oluştur
            let emotionBars = '';
            
            // Duyguları skorlarına göre sırala
            const sortedEmotions = [...result.emotions].sort((a, b) => b.score - a.score);
            
            sortedEmotions.forEach(emotion => {
                const percentage = (emotion.score * 100).toFixed(2);
                emotionBars += `
                <div class="emotion-bar">
                    <div class="emotion-fill" style="width: ${percentage}%; background-color: ${emotion.color};"></div>
                    <div class="emotion-label">${emotion.emotion}</div>
                    <div class="emotion-score">${percentage}%</div>
                </div>`;
            });
            
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Duygu Analizi Sonuçları
                </div>
                <div class="result-content">
                    <p><strong>Metin:</strong> "${result.text}"</p>
                    <hr style="margin: 1rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                    <div>${emotionBars}</div>
                </div>
            </div>`;
        });
    });
    
    // Konuşma Tanıma
    const konusmaButton = document.getElementById('konusma-button');
    const konusmaOutput = document.getElementById('konusma-output');
    const audioInput = document.getElementById('audio-input');
    const audioPreview = document.getElementById('audio-preview');
    const recordButton = document.getElementById('record-button');
    const stopButton = document.getElementById('stop-button');
    
    // Ses kaydı için değişkenler
    let mediaRecorder;
    let audioChunks = [];
    let audioBlob;
    
    // Kayıt başlat
    recordButton.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (e) => {
                audioChunks.push(e.data);
            };
            
            mediaRecorder.onstop = () => {
                audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPreview.src = audioUrl;
                audioPreview.style.display = 'block';
                
                // Dosya girişini temizle
                audioInput.value = '';
            };
            
            // Kaydı başlat
            audioChunks = [];
            mediaRecorder.start();
            
            // Buton durumlarını güncelle
            recordButton.disabled = true;
            stopButton.disabled = false;
            
        } catch (err) {
            console.error('Mikrofon erişimi hatası:', err);
            konusmaOutput.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-negative">Hata</div>
                <div class="result-content">Mikrofon erişimi sağlanamadı: ${err.message}</div>
            </div>`;
        }
    });
    
    // Kaydı durdur
    stopButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            
            // Medya akışını kapat
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            // Buton durumlarını güncelle
            recordButton.disabled = false;
            stopButton.disabled = true;
        }
    });
    
    konusmaButton.addEventListener('click', () => {
        const dil = document.querySelector('input[name="konusma-dil"]:checked').value;
        let audioFile;
        
        // Dosya girişinden veya kayıttan ses dosyasını al
        if (audioInput.files && audioInput.files[0]) {
            audioFile = audioInput.files[0];
        } else if (audioBlob) {
            audioFile = new File([audioBlob], "recorded_audio.wav", { type: "audio/wav" });
        } else {
            konusmaOutput.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-negative">Hata</div>
                <div class="result-content">Lütfen bir ses dosyası yükleyin veya kaydedin!</div>
            </div>`;
            return;
        }
        
        const formData = new FormData();
        formData.append('audio', audioFile);
        formData.append('dil', dil);
        
        sendFileRequest('konusma-tanima', formData, konusmaOutput, (result, outputElement) => {
            // Zaman damgalı transkripsiyon
            let timestampsHtml = '';
            if (result.timestamps && result.timestamps.length > 0) {
                timestampsHtml = '<div style="margin-top: 1rem;"><strong>Zaman Damgalı Transkripsiyon:</strong><ul>';
                
                result.timestamps.forEach(ts => {
                    timestampsHtml += `<li><strong>${ts.start_time} - ${ts.end_time}:</strong> ${ts.text}</li>`;
                });
                
                timestampsHtml += '</ul></div>';
            }
            
            outputElement.innerHTML = `<div class="result-card">
                <div class="result-header sentiment-neutral">
                    Konuşma Tanıma (${result.language})
                </div>
                <div class="result-content">
                    <p><strong>Transkripsiyon:</strong></p>
                    <p>${result.text}</p>
                    ${timestampsHtml}
                </div>
            </div>`;
        });
    });
});