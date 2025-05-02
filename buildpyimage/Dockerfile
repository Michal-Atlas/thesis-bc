FROM fedora

RUN dnf install -y python311 which

RUN ln -s /usr/bin/python3.11 /usr/bin/python3

WORKDIR /root

COPY ./enter.sh /enter.sh

ENTRYPOINT /enter.sh
