from .s3 import s3a_path_with_auth


def spark_redshift_options(   # https://github.com/databricks/spark-redshift
        options,
        access_key_id=None, secret_access_key=None):
    options['tempdir'] = \
        s3a_path_with_auth(
            s3_path=options['tempdir'],
            access_key_id=access_key_id,
            secret_access_key=secret_access_key)

    options['forward_spark_s3_credentials'] = True

    options['driver'] = 'com.amazon.redshift.jdbc.Driver'

    # *** BUG: CSV FORMAT DOESN'T WORK ***
    # if 'tempformat' not in options:
    #     options['tempformat'] = 'CSV GZIP'

    return options
