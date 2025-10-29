base_url = "http://10.243.65.197:12004"

GET /api/v2/preupload 预上传
说明：上传链接10分钟有效
请求示例
curl -X GET 'http://localhost:12004/api/v2/preupload'
返回示例
{
  "code": 0,
  "data": {
    "timestamp": 1753078573159,
    "uid": "250b8859-1d3b-4d48-8f0e-fd56e13e5a92",
    "url": "https://glm-data-ocr-data.cn-beijing-vpc.oss.aliyuncs.com/services%2Fmaas%2Fdocs%2F%2F250b8859-1d3b-4d48-8f0e-fd56e13e5a92?Expires=1753082173&OSSAccessKeyId=LTAI5tSQF36yg2CAeunPrmLk&Signature=XvnHUi3A3Iqg3YpLH4DcQ2kpVmk%3D"
  },
  "msg": "success"
}
POST /api/v2/convert/parse 异步解析
请求字段
字段
类型
位置
说明
doc_type
string
body
取值：pdf/img
图片支持：
  JPEG (.jpg, .jpeg)
  PNG (.png)
uid
string
body
preupload获取
ocr_all
bool
body
true: 全部使用ocr，false：先使用pdf工具提取文字，提取不到的使用ocr提取
请求示例
curl -X POST 'http://localhost:12004/api/v2/convert/parse' -d '{"doc_type":"pdf","uid":"ac55e386-2208-4a2d-af27-bd9f5ffac922"}'
返回示例
{
    "code": 0,
    "msg": "success"
}
POST /api/v2/upload/parse 上传&解析
说明：支持两种方式，文件和url
注意：直接上传文件会占用带宽，仅限小文件
请求字段
字段
类型
位置
说明
doc_type
string
body
取值：pdf/img
图片支持：
  JPEG (.jpg, .jpeg)
  PNG (.png)
ocr_all
bool
body
true: 全部使用ocr，false：先使用pdf工具提取文字，提取不到的使用ocr提取
url
string
body
可以get下载的文档链接
file
file/binary
body
需要上传的文档文件
请求示例一：发送JSON格式（application/json）
curl -X POST 'http://localhost:12004/api/v2/upload/parse' \
  -H 'Content-Type: application/json' \
  -d '{
    "doc_type": "pdf",
    "ocr_all": false,
    "url": "https://glm-data-ocr-data.oss-cn-beijing.aliyuncs.com/misc/physics/0717_Physics/cap/1994_CAP_exam.pdf?OSSAccessKeyId=LTAI5tSQF36yg2CAeunPrmLk&Expires=1753350424&Signature=NE5YPpmmlpo4SM4xsCkCCD5NybY%3D"
  }'
请求示例二：上传文件（multipart/form-data）
curl -X POST 'http://localhost:12004/api/v2/upload/parse' \
  -F "file=@/root/workdir/cpplibs/data/demo/test/1-1.pdf" \
  -F "doc_type=pdf" \
  -F "ocr_all=false"
返回示例
{
  "code": 0,
  "data": {
    "uid": "dbbabe78-f844-43b2-9a05-334bee6abb01"
  },
  "msg": "success"
}
GET /api/v2/convert/result 结果查询
说明：结果链接保留24小时
请求示例
curl -X GET 'http://localhost:12004/api/v2/convert/result?uid=ac55e386-2208-4a2d-af27-bd9f5ffac922'
返回示例
{
  "code": 0,
  "data": {
    "page_count": 11,
    "status": "success",
    "url": "https://glm-data-ocr-data.oss-cn-beijing.aliyuncs.com/services%2Fmaas%2Fpa%2F09bf0713-d533-4a16-94be-0a692faa4e90.tar?Expires=1753337559&OSSAccessKeyId=LTAI5tSQF36yg2CAeunPrmLk&Signature=OrWmSwCjByAR6AMtxWQZ5h2P%2BXQ%3D"
  },
  "msg": "success"
}
GET api/v2/status 状态查询
说明：查询目前队列情况,
请求示例
curl http://localhost:12004/api/v2/status
返回示例
{
  "code": 0,
  "data": {
    "docs": 0,
    "formula": 0,
    "layout_det": 0,
    "mfd": 0,
    "ocr_cls_rec": 0,
    "ocr_det": 0,
    "table_structure": 0
  },
  "msg": "success"
}
返回字段说明
名称
说明
docs
队列中排队文档
formula
公式模型排队任务
layout_det
布局模型排队任务
mfd
公式检测排队任务
ocr_cls_rec
ocr识别排队任务
ocr_det
ocr探测排队任务
table_structure
表格结构排队任务
GET /health 健康检查
请求示例
curl http://localhost:12004/health
返回示例
ok
调用示例
示例一  [4步：预上传->上传->解析->结果]
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
    local_file_path = "/root/workdir/data/glm_ocr/ch_pdf/曾谨言《量子力学教程》（第3版）笔记和课后习题（含考研真题）详解.pdf"
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
示例二 [2步: 上传&解析->结果]
import json
import time
import requests as rq

base_url = "http://localhost:12004"

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
    file_path = "/root/workdir/cpplibs/data/demo/test/1-1.pdf"
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
状态码
注意：http返回200后，通过code确定是否解析成功
code（错误码）
msg（消息）
0
success
4001
invalid parameter
4003
oss gen upload url failed
4004
invalid doc type
4005
invalid uid
4006
invalid ocr all
4007
oss file not found
4008
msg push failed
4009
msg set failed
4010
request not contain uid
4011
uid is empty
4010
uid not found
4011
oss gen download url failed
4012
oss object meta failed
4013
oss object meta not found code
4014
oss object meta not found msg
4015
oss object meta not found page count
4016
not upload file
4017
invalid file url
4018
invalid content type
4019
use url download file failed
4020
oss upload failed
5001
src_path or dst_path is empty
5002
local path not found
5004
oss download failed
5005
oss tar info invalid
5006
plugin not found
5007
plugin timeout
5008
plugin base_info rect invalid
5009
plugin base_info pix invalid
5010
plugin invalid img
5011
plugin img decode failed
5012
plugin unknown error
5013
incomplete page
5999
unknown error
-1
internal server error
压力测试
- 随机采样500张图片和500个pdf压测1小时，未发现问题

评测
自评参考： Glm ocr benmark量化评估
agent评测参考：知识库-自研ocr效果评估
maas评测参考：文件解析三方工具需求

结果字段说明
1. 返回结果包含三个文件和一个文件夹，如下图
  1. imgs图片存储位置
  2. layout.json 页面完整布局信息
  3. res.md 解析结果markdown
  4. res.txt解析结果文本
[图片]
字段
类型
说明
示例
code
int
返回码
0
doc_size
int
文档大小（字节）
223722
doc_type
string
文档类型
"pdf"
dst_path
string
目标路径
""
markdown
string
markdown内容
""
msg
string
消息
""
ocr_all
bool
是否全局OCR
false
page_count
int
页数
1
pages
array
页面列表

pages.page_no
int
页码（从0开始）
0
pages.height
int
页面高度
783
pages.width
int
页面宽度
612
pages.scale
float
坐标缩放比例
2.616
pages.abandon_blocks
array
弃用块

pages.abandon_blocks.bbox
object
弃用块边界框
{x0:177,x1:1454,y0:1975,y1:2047}
pages.abandon_blocks.conf
float
弃用块置信度
0.8105
pages.abandon_blocks.font_size
float
弃用块字体大小
0.0
pages.abandon_blocks.label
string
弃用块标签
"Abandon"
pages.abandon_blocks.label_id
int
弃用块标签ID
2
pages.abandon_blocks.lines
array
行内容

pages.abandon_blocks.lines.bbox
object
行边界框
{x0:178,x1:1418,y0:2014,y1:2044}
pages.abandon_blocks.lines.font_size
float
行字体大小
0.0
pages.abandon_blocks.lines.text
string
行文本内容

pages.abandon_blocks.source
string
来源
"layout det"
pages.abandon_blocks.text
string
块文本内容
""
pages.blocks
array
内容块

pages.blocks.bbox
object
块边界框

pages.blocks.conf
float
块置信度

pages.blocks.font_size
float
块字体大小

pages.blocks.label
string
块标签（如Title、Text）

pages.blocks.label_id
int
块标签ID

pages.blocks.lines
array
块中的行

pages.blocks.lines.bbox
object
行边界框

pages.blocks.lines.font_size
float
行字体大小

pages.blocks.lines.text
string
行文本

pages.blocks.source
string
来源

pages.blocks.text
string
块全部文本

pages.formula_dets
array
公式检测结果
[]
pages.layout_dets
array
布局检测结果

pages.layout_dets.bbox
object
边界框

pages.layout_dets.conf
float
置信度

pages.layout_dets.label
string
标签

pages.layout_dets.label_id
int
标签id

pages.ocr_dets
array
OCR结果

pages.ocr_dets.poly
array
多边形坐标（如4点）
[[178,2014],[1418,2016],[1418,2044],[178,2042]]
pages.ocr_dets.score
float
置信度
0.6318
pages_success_ratio
float
页面成功识别比例
0.0
src_path
string
原文件路径
""
text
string
文档文本内容
""