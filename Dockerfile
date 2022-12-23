FROM python:3.8.16-slim
#===============================#
# Docker Image Configuration	#
#===============================#
LABEL org.opencontainers.image.source='https://github.com/eipm/stork-v' \
    vendor='Englander Institute for Precision Medicine' \
    description='STORK-V' \
    maintainer='paz2010@med.cornell.edu' \
    base_image='python' \
    base_image_version='3.8.16-slim'

ENV APP_NAME='stork-v' \
    TZ='US/Eastern'

#===================================#
# Install Prerequisites             #
#===================================#
COPY requirements.txt /${APP_NAME}/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip && pip install -r /${APP_NAME}/requirements.txt
#===================================#
# Copy Files and set work directory	#
#===================================#
COPY src /${APP_NAME}
WORKDIR /${APP_NAME}
#===================================#
# Startup                           #
#===================================#
EXPOSE 80
ENV PATH=$PATH:/${APP_NAME}
ENV PYTHONPATH /${APP_NAME}
VOLUME /${APP_NAME}/api/data
VOLUME /${APP_NAME}/stork_v/models

HEALTHCHECK --interval=30s --timeout=30s --retries=3 \
    CMD curl -f -k http://0.0.0.0/api/healthcheck || exit 1

CMD python3 /${APP_NAME}/main.py
