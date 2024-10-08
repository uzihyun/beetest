function loadImagePreview(event) {
            const file = event.target.files[0];
            const preview = document.getElementById('preview');

            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block'; // 이미지가 선택되면 미리보기 표시
                }
                reader.readAsDataURL(file);
            } else {
                preview.src = '';
                preview.style.display = 'none'; // 이미지가 없을 경우 미리보기 숨김
            }
        }