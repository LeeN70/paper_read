import json
import time
import requests as rq

base_url = "http://10.243.65.197:12004"

session = rq.Session()

# 文件+参数    
def upload_parse_by_file(file_path, doc_type="pdf", ocr_all=False):
    url = f"{base_url}/api/v2/upload/parse"
    files = {'file': open(file_path, 'rb')}
    data = {
        "doc_type": doc_type,
        "ocr_all": ocr_all,
    }
    res = session.post(url, files=files, data=data)
    return res.json()
   
# url + 参数
def upload_parse_by_url(file_url, doc_type="pdf", ocr_all=False):
    url = f"{base_url}/api/v2/upload/parse"
    data = {
        "url": file_url,
        "doc_type": doc_type,
        "ocr_all": ocr_all,
    }
    res = session.post(url, json=data)
    return res.json()
    
def result(uid):
    url = f"{base_url}/api/v2/convert/result?uid={uid}"
    res = session.get(url)
    if res.status_code != 200:
        print(f"result failed, status_code: {res.status_code}, msg: {res.text}")
        return {}
    return res.json()

if __name__ == "__main__":
    file_path = "/data/lixin/paper-reader-3/cache_zai/1706.03762/1706.03762.pdf"
    url = "https://glm-data-ocr-data.oss-cn-beijing.aliyuncs.com/misc/physics/0717_Physics/cap/1994_CAP_exam.pdf?OSSAccessKeyId=LTAI5tSQF36yg2CAeunPrmLk&Expires=1753350424&Signature=NE5YPpmmlpo4SM4xsCkCCD5NybY%3D"
    doc_type = "pdf"
    ocr_all = True
    
    # step 1: 上传&解析
    upload_info = upload_parse_by_file(file_path, doc_type, ocr_all)
    # upload_info = upload_parse_by_url(url, doc_type, ocr_all)
    print(json.dumps(upload_info, ensure_ascii=False, indent=2))
    
    # step 2: 轮训结果
    while True:
        result_info = result(upload_info["data"]["uid"])
        if result_info["code"] != 0:
            print(json.dumps(result_info, ensure_ascii=False, indent=2))
            break
        
        if result_info["data"]["status"] == "success":
            print(json.dumps(result_info, ensure_ascii=False, indent=2))
            break
       
        time.sleep(0.01)