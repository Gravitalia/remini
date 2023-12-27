FROM rust:alpine3.18 AS builder

RUN USER=root cargo new --bin remini
WORKDIR /remini

ENV     RUSTFLAGS="-C target-feature=-crt-static"
RUN     apk add -q --update-cache --no-cache build-base openssl-dev musl pkgconfig protobuf-dev

COPY . .

RUN cargo build --release

FROM alpine:3.18 AS runtime

RUN apk add --no-cache libgcc

RUN addgroup -S appgroup && adduser -S rust -G appgroup
USER rust

# Copy binary.
COPY --from=builder /remini/target/release/remini /bin/remini
# Copy ML models.
COPY --from=builder /remini/src/corpus/model.onnx /src/corpus/model.onnx

EXPOSE 50051/tcp
CMD     ["./bin/remini"]
