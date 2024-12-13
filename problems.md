# Tagging api not found
requests.exceptions.ConnectionError: HTTPConnectionPool(host='tagging', port=8000): Max retries exceeded with url: /process_audio (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f5b62549ca0>: Failed to establish a new connection: [Errno -2] Name or service not known'))
Traceback:
File "/app/streamlit_app.py", line 44, in <module>
    tagging_response = requests.post(tagging_api_url, files=files, data={"filename": uploaded_file.name})
File "/usr/local/lib/python3.9/site-packages/requests/api.py", line 115, in post
    return request("post", url, data=data, json=json, **kwargs)
File "/usr/local/lib/python3.9/site-packages/requests/api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
File "/usr/local/lib/python3.9/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
File "/usr/local/lib/python3.9/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
File "/usr/local/lib/python3.9/site-packages/requests/adapters.py", line 700, in send
    raise ConnectionError(e, request=request)

# Opensearch API not found
