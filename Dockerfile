ARG PG_MAJOR=12.10
FROM postgres:$PG_MAJOR
ARG PG_MAJOR

COPY . /tmp/pgdl

RUN sed -i "s@http://\(deb\|security\).debian.org@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
		apt-get update && \
		apt-get install -y --no-install-recommends build-essential postgresql-server-dev-12 cmake wget git pkg-config libgoogle-perftools-dev unzip ca-certificates && \
		cd /tmp/pgdl/ && \
		bash /tmp/pgdl/shell/download_build_thirdparty.sh && \
		mkdir -p build && \
		cd build && \
		cmake -DCMAKE_PREFIX_PATH="/tmp/pgdl/third_party/libtorch/" .. && \
		make clean && \
		make -j$(nproc)  && \
		make install && \
		apt-get remove -y build-essential postgresql-server-dev-12 cmake wget git pkg-config libgoogle-perftools-dev unzip ca-certificates && \
		apt-get autoremove -y && \
		rm -rf /var/lib/apt/lists/*

