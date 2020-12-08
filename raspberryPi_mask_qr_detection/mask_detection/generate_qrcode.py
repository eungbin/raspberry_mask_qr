import qrcode

# Hello World! 로 QR 코드 생성
img = qrcode.make('Hello World!')

# 생성된 이미지를 helloworld_qrcode.png로 저장
img.save('generate_code_test.png')