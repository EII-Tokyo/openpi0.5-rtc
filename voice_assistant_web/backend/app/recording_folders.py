"""Proxy collection folder management; never sends robot commands."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, constr

router = APIRouter()
SERVICE = os.getenv('ALOHA_TASK_WEB_INTERNAL_URL', 'http://127.0.0.1:8765').rstrip('/')
EXPECTED_ROOT = os.getenv('ALOHA_COLLECTION_HOST_ROOT', '/home/eii/aloha-2.0/aloha_data/aloha_stationary')


class FolderRequest(BaseModel):
    name: constr(min_length=1, max_length=96)


def request_folders(name=None):
    try:
        # Fail closed before creation if collection and recorder roots differ.
        with urlopen(SERVICE + '/api/recording-folders', timeout=5) as response:
            result = json.load(response)
        if result.get('root') != EXPECTED_ROOT:
            raise HTTPException(409, '采集页面和录制保存目录不一致，请检查配置')
        if name is not None:
            request = Request(SERVICE + '/api/recording-folders',
                              data=json.dumps({'name': name}).encode(),
                              headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=5) as response:
                result = json.load(response)
        return result
    except HTTPError as exc:
        if exc.code == 409:
            raise HTTPException(409, '名称已存在，请刷新并选择已有文件夹') from None
        if exc.code == 400:
            raise HTTPException(400, '文件夹名称无效，请使用字母、数字、下划线、点或横线') from None
        raise HTTPException(502, '数据采集页面目录服务异常') from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise HTTPException(503, '无法连接数据采集页面，请检查服务后刷新') from None


@router.get('/api/recording/folders')
def list_folders():
    return request_folders()


@router.post('/api/recording/folders')
def create_folder(request: FolderRequest):
    return request_folders(request.name)
