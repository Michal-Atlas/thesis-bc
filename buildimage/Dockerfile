FROM ubuntu:22.04

RUN apt -o APT::Sandbox::User=root update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -o APT::Sandbox::User=root install -y gawk wget git diffstat unzip texinfo gcc build-essential chrpath socat cpio python3 python3-pip python3-pexpect xz-utils debianutils iputils-ping python3-git python3-jinja2 python3-subunit zstd liblz4-tool file locales libacl1 tmux
RUN locale-gen en_US.UTF-8

WORKDIR /src/ci-meta-tq

COPY ./enter.sh /enter.sh

ENTRYPOINT /enter.sh
