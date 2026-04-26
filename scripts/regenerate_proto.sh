#!/usr/bin/env bash
set -e
protoc -I proto \
    --go_out=gateway/gen \
    --go-grpc_out=gateway/gen \
    --go_opt=paths=source_relative \
    --go-grpc_opt=paths=source_relative \
    proto/imprimer.proto

python -m grpc_tools.protoc \
    -I proto \
    --python_out=engine \
    --grpc_python_out=engine \
    proto/imprimer.proto
