import botocore
import boto3
import configparser
import os
import warnings


CREDS_FILE_PATH = os.path.expanduser('~/.aws/credentials')

if os.path.isfile(CREDS_FILE_PATH):
    CREDS = configparser.ConfigParser()
    CREDS.read_file(open(CREDS_FILE_PATH))

else:
    CREDS = None


_CLIENTS = {}


def key_pair(profile='default'):
    if CREDS:
        if profile in CREDS:
            return CREDS[profile]['aws_access_key_id'], \
                CREDS[profile]['aws_secret_access_key']

        elif (profile != 'default') and ('default' in CREDS):
            warnings.warn(
                message='*** "{}" PROFILE DOES NOT EXIST in "{}"; FALLING BACK TO DEFAULT PROFILE ***'
                    .format(profile, CREDS_FILE_PATH))

            return CREDS['default']['aws_access_key_id'], \
               CREDS['default']['aws_secret_access_key']

        else:
            warnings.warn(
                message='*** NEITHER "{}" NOR "default" PROFILE EXISTS IN "{}"; FALLING BACK TO EC2 INSTANCE PROFILE ***'
                    .format(profile, CREDS_FILE_PATH))

            return None, None

    else:
        warnings.warn(
            message='*** "{}" DOES NOT EXIST; FALLING BACK TO EC2 INSTANCE PROFILE ***'
                .format(CREDS_FILE_PATH))

        return None, None


def client(service, access_key_id=None, secret_access_key=None):
    global _CLIENTS

    tup = service, access_key_id, secret_access_key

    if tup not in _CLIENTS:
        _CLIENTS[tup] = \
            boto3.client(
                service,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=botocore.client.Config(
                    connect_timeout=9,
                    read_timeout=9))

    return _CLIENTS[tup]
