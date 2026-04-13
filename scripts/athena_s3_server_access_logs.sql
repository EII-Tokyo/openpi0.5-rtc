-- S3 服务器访问日志 -> Athena：按桶汇总 bytes_sent（出站字节，见日志字段 bytessent）
-- 日志投递桶（与 CLI 配置一致）: s3://eii-s3-server-access-logs-980259683008/
-- 在 Athena 控制台选择工作组/结果桶后执行；LOCATION 勿加尾部通配符。

CREATE DATABASE IF NOT EXISTS s3_access_logs;

-- 官方 RegexSerDe，见:
-- https://docs.aws.amazon.com/athena/latest/ug/s3-server-access-logs.html
CREATE EXTERNAL TABLE IF NOT EXISTS s3_access_logs.tokyo_buckets_s3_access (
  bucketowner STRING,
  bucket STRING,
  requestdatetime STRING,
  remoteip STRING,
  requester STRING,
  requestid STRING,
  operation STRING,
  key STRING,
  requesturi STRING,
  httpstatus STRING,
  errorcode STRING,
  bytessent BIGINT,
  objectsize BIGINT,
  totaltime STRING,
  turnaroundtime STRING,
  referer STRING,
  useragent STRING,
  versionid STRING,
  hostid STRING,
  sigv4signed STRING,
  ttlssession STRING,
  ttlscipher STRING,
  hostheader STRING,
  requesttype STRING,
  requestpriority STRING,
  requestttl STRING,
  additionalerror STRING,
  accesspointarn STRING,
  operationcount INT,
  bucketregion STRING,
  srcbucket STRING,
  objectlockretainuntildate STRING,
  objectlocklegalholdstatus STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
WITH SERDEPROPERTIES (
  'serialization.format' = '1',
  'input.regex' = '([^ ]*) ([^ ]*) \\[(.*?)\\] ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) (\"[^\"]*\"|-) (-|[0-9]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) (\"[^\"]*\"|-) ([^ ]*)(?: ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) (\".*?\") ([^ ]*)(?: ([^ ]*) ([^ ]*))?(?: ([^ ]*) ([^ ]*))?.*$'
)
STORED AS TEXTFILE
LOCATION 's3://eii-s3-server-access-logs-980259683008/'
TBLPROPERTIES ('has_encrypted_data' = 'false');

-- 按「被访问的桶」汇总传出字节（REST.GET.OBJECT 等读对象）
-- 新启用日志后需等待数小时才有数据。
SELECT
  bucket,
  SUM(bytessent) AS total_bytes_sent,
  ROUND(SUM(bytessent) / 1024 / 1024 / 1024, 3) AS total_gib
FROM s3_access_logs.tokyo_buckets_s3_access
WHERE operation LIKE 'REST.GET%'
  AND bytessent IS NOT NULL
GROUP BY 1
ORDER BY total_bytes_sent DESC;

-- 按桶 + 对象前缀（便于看热点目录）
SELECT
  bucket,
  SPLIT_PART(key, '/', 1) AS top_prefix_segment,
  SUM(bytessent) AS total_bytes_sent
FROM s3_access_logs.tokyo_buckets_s3_access
WHERE operation LIKE 'REST.GET%'
  AND bytessent IS NOT NULL
GROUP BY 1, 2
ORDER BY total_bytes_sent DESC
LIMIT 100;
