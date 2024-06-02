# Add necessary group definitions
groupadd -g 1049800222 group_1049800222 || true
groupadd -g 998 group_998 || true
groupadd -g 999 group_999 || true
groupadd -g 6000 group_6000 || true
groupadd -g 1625200002 group_1625200002 || true
groupadd -g 1625200035 group_1625200035 || true
groupadd -g 1625200036 group_1625200036 || true

python python /nethome/hhwang/kagaku/coconut_100.py

