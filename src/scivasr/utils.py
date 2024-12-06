import hashlib
import json
import os

def get_file_hash(file_name: str):
    return hashlib.sha1(file_name.encode()).hexdigest()[:8]


write_json = lambda data, file_name: open(file_name, "w").write(
    json.dumps(data, ensure_ascii=False)
)

write_file = lambda data, file_name: open(file_name, "w").write(data)

write_lines = lambda lines, file_name: open(file_name, "w").write(
    "\n".join(lines) + "\n"
)

read_json = lambda file_name: json.loads(open(file_name).read())

read_file = lambda file_name: open(file_name).read()

check_existence = lambda file_name: os.path.exists(file_name)