NOISE_API_URL = "https://www.noiseinfo.or.kr/getNoiseData.dojson"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "https://www.noiseinfo.or.kr/index.jsp",
}

TARGET_REGION = "서울특별시 강남구"

KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
KAFKA_TOPIC_RAW = "noise_raw_data"
