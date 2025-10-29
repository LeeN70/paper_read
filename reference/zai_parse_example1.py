import json
import time
import requests as rq

base_url = "http://10.243.65.197:12004"

session = rq.Session()
    
def preupload():
    url = f"{base_url}/api/v2/preupload"
    res = session.get(url)
    if res.status_code != 200:
        print(f"preupload failed, status_code: {res.status_code}, msg: {res.text}")
        return
    return res.json()

def upload(url, file_path):
    with open(file_path, "rb") as f:
        res = session.put(url, data=f)
    if res.status_code != 200:
        print(f"upload failed, status_code: {res.status_code}, msg: {res.text}")
        return
    
def parse(uid, doc_type="pdf"):
    url = f"{base_url}/api/v2/convert/parse"
    data = {
        "uid": uid,
        "doc_type": doc_type,
    }
    res = session.post(url, data=json.dumps(data))
    
def result(uid):
    url = f"{base_url}/api/v2/convert/result?uid={uid}"
    res = session.get(url)
    if res.status_code != 200:
        print(f"result failed, status_code: {res.status_code}, msg: {res.text}")
        return {}
    return res.json()

if __name__ == "__main__":
    local_file_path = "/data/lixin/paper-reader-3/cache_zai/1706.03762/1706.03762.pdf"
    doc_type = "pdf"
    
    # step 1: 获取上传文件的url
    pre_info = preupload()
    print(json.dumps(pre_info, ensure_ascii=False, indent=2))
    
    # step 2: 上传文件
    upload(pre_info["data"]["url"], local_file_path)
    
    # step 3: 触发异步解析
    parse(pre_info["data"]["uid"], doc_type)
    
    # step 4: 轮训结果
    while True:
        result_info = result(pre_info["data"]["uid"])
        if result_info["code"] != 0:
            print(json.dumps(result_info, ensure_ascii=False, indent=2))
            break
        
        if result_info["data"]["status"] == "success":
            print(json.dumps(result_info, ensure_ascii=False, indent=2))
            break
       
        time.sleep(0.01)