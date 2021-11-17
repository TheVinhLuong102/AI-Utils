from pandas import DataFrame


# ref: https://aws.amazon.com/ec2/instance-types


N_CPUS_KEY = 'n_cpus'
MEMORY_GiB_KEY = 'memory_gib'

N_FPGAS_KEY = 'n_fpgas'

N_GPUS_KEY = 'n_gpus'
GPU_MEMORY_GiB_KEY = 'gpu_memory_gib'
GPU_P2P_KEY = 'gpu_p2p'

STORAGE_KEY = 'storage'
STORAGE_GB_KEY = 'storage_gb'
EBS_BANDWIDTH_Gbps_KEY = 'ebs_bandwidth_gbps'
NETWORK_Gbps_KEY = 'network_gbps'


INSTANCE_TYPES_INFO = \
    DataFrame.from_dict(
        data={
            # COMPUTE-OPTIMIZED
            # with Enhanced Networking
            'c5n.large': {
                N_CPUS_KEY: 2,
                MEMORY_GiB_KEY: 5.25,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: '<= 3.5',

                NETWORK_Gbps_KEY: '<= 25'
            },

            'c5n.xlarge': {
                N_CPUS_KEY: 4,
                MEMORY_GiB_KEY: 10.5,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: '<= 3.5',

                NETWORK_Gbps_KEY: '<= 25'
            },

            'c5n.2xlarge': {
                N_CPUS_KEY: 8,
                MEMORY_GiB_KEY: 21,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: '<= 3.5',

                NETWORK_Gbps_KEY: '<= 25'
            },

            'c5n.4xlarge': {
                N_CPUS_KEY: 16,
                MEMORY_GiB_KEY: 42,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: 3.5,

                NETWORK_Gbps_KEY: '<= 25'
            },

            'c5n.9xlarge': {
                N_CPUS_KEY: 36,
                MEMORY_GiB_KEY: 96,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: 7,

                NETWORK_Gbps_KEY: 50
            },

            'c5n.18xlarge': {
                N_CPUS_KEY: 72,
                MEMORY_GiB_KEY: 192,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: 14,

                NETWORK_Gbps_KEY: 100
            },

            'c5n.metal': {
                N_CPUS_KEY: 72,
                MEMORY_GiB_KEY: 192,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: 14,

                NETWORK_Gbps_KEY: 100
            },

            # ACCELERATED COMPUTING
            # FPGA
            'f1.2xlarge': {
                N_CPUS_KEY: 8,
                MEMORY_GiB_KEY: 122,

                N_FPGAS_KEY: 1,

                STORAGE_KEY: 'SSD',
                STORAGE_GB_KEY: 470,

                NETWORK_Gbps_KEY: '<= 10'
            },

            'f1.4xlarge': {
                N_CPUS_KEY: 16,
                MEMORY_GiB_KEY: 244,

                N_FPGAS_KEY: 2,

                STORAGE_KEY: 'SSD',
                STORAGE_GB_KEY: 940,

                NETWORK_Gbps_KEY: '<= 10'
            },

            'f1.16xlarge': {
                N_CPUS_KEY: 64,
                MEMORY_GiB_KEY: 976,

                N_FPGAS_KEY: 8,

                STORAGE_KEY: 'SSD',
                STORAGE_GB_KEY: 3760,

                NETWORK_Gbps_KEY: 25
            },

            # GPU
            'g3s.xlarge': {
                N_CPUS_KEY: 4,
                MEMORY_GiB_KEY: 30.5,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 8,

                NETWORK_Gbps_KEY: '<= 10'
            },

            'g3.4xlarge': {
                N_CPUS_KEY: 16,
                MEMORY_GiB_KEY: 122,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 8,

                NETWORK_Gbps_KEY: '<= 10'
            },

            'g3.8xlarge': {
                N_CPUS_KEY: 32,
                MEMORY_GiB_KEY: 244,

                N_GPUS_KEY: 2,
                GPU_MEMORY_GiB_KEY: 16,

                NETWORK_Gbps_KEY: 10
            },

            'g3.16xlarge': {
                N_CPUS_KEY: 64,
                MEMORY_GiB_KEY: 488,

                N_GPUS_KEY: 4,
                GPU_MEMORY_GiB_KEY: 32,

                NETWORK_Gbps_KEY: 25
            },

            'g4dn.xlarge': {
                N_CPUS_KEY: 4,
                MEMORY_GiB_KEY: 16,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 16,

                STORAGE_GB_KEY: 125,

                NETWORK_Gbps_KEY: '<= 25'
            },

            'g4dn.2xlarge': {
                N_CPUS_KEY: 8,
                MEMORY_GiB_KEY: 32,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 16,

                STORAGE_GB_KEY: 225,

                NETWORK_Gbps_KEY: '<= 25'
            },

            'g4dn.4xlarge': {
                N_CPUS_KEY: 16,
                MEMORY_GiB_KEY: 64,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 16,

                STORAGE_GB_KEY: 225,

                NETWORK_Gbps_KEY: '<= 25'
            },

            'g4dn.8xlarge': {
                N_CPUS_KEY: 32,
                MEMORY_GiB_KEY: 128,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 16,

                STORAGE_GB_KEY: 900,

                NETWORK_Gbps_KEY: 50
            },

            'g4dn.16xlarge': {
                N_CPUS_KEY: 64,
                MEMORY_GiB_KEY: 256,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 16,

                STORAGE_GB_KEY: 900,

                NETWORK_Gbps_KEY: 50
            },

            'g4dn.12xlarge': {
                N_CPUS_KEY: 48,
                MEMORY_GiB_KEY: 192,

                N_GPUS_KEY: 4,
                GPU_MEMORY_GiB_KEY: 64,

                STORAGE_GB_KEY: 900,

                NETWORK_Gbps_KEY: 50
            },

            'g4dn.metal': {
                N_CPUS_KEY: 96,
                MEMORY_GiB_KEY: 384,

                N_GPUS_KEY: 8,
                GPU_MEMORY_GiB_KEY: 128,

                STORAGE_GB_KEY: 1800,

                NETWORK_Gbps_KEY: 100
            },

            'p2.xlarge': {
                N_CPUS_KEY: 4,
                MEMORY_GiB_KEY: 61,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 12,

                NETWORK_Gbps_KEY: 'High'
            },

            'p2.8xlarge': {
                N_CPUS_KEY: 32,
                MEMORY_GiB_KEY: 488,

                N_GPUS_KEY: 8,
                GPU_MEMORY_GiB_KEY: 96,

                NETWORK_Gbps_KEY: 10
            },

            'p2.16xlarge': {
                N_CPUS_KEY: 64,
                MEMORY_GiB_KEY: 732,

                N_GPUS_KEY: 16,
                GPU_MEMORY_GiB_KEY: 192,

                NETWORK_Gbps_KEY: 25
            },

            'p3.2xlarge': {
                N_CPUS_KEY: 8,
                MEMORY_GiB_KEY: 61,

                N_GPUS_KEY: 1,
                GPU_MEMORY_GiB_KEY: 16,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: 1.5,

                NETWORK_Gbps_KEY: '<= 10'
            },

            'p3.8xlarge': {
                N_CPUS_KEY: 32,
                MEMORY_GiB_KEY: 244,

                N_GPUS_KEY: 4,
                GPU_MEMORY_GiB_KEY: 64,
                GPU_P2P_KEY: 'NVLink',

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: 7,

                NETWORK_Gbps_KEY: 10
            },

            'p3.16xlarge': {
                N_CPUS_KEY: 64,
                MEMORY_GiB_KEY: 488,

                N_GPUS_KEY: 8,
                GPU_MEMORY_GiB_KEY: 128,
                GPU_P2P_KEY: 'NVLink',

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: 14,

                NETWORK_Gbps_KEY: 25
            },

            'p3dn.24xlarge': {
                N_CPUS_KEY: 96,
                MEMORY_GiB_KEY: 768,

                N_GPUS_KEY: 8,
                GPU_MEMORY_GiB_KEY: 256,
                GPU_P2P_KEY: 'NVLink',

                STORAGE_KEY: 'NVMe SSD',
                STORAGE_GB_KEY: 1800,
                EBS_BANDWIDTH_Gbps_KEY: 14,

                NETWORK_Gbps_KEY: 100
            },

            'x1e.32xlarge': {
                N_CPUS_KEY: 128,
                MEMORY_GiB_KEY: 3904,

                STORAGE_KEY: 'EBS-Only',
                EBS_BANDWIDTH_Gbps_KEY: '',

                NETWORK_Gbps_KEY: 25
            }
        },
        orient='index',
        dtype=None,
        columns=
            (N_CPUS_KEY,
             MEMORY_GiB_KEY,

             N_FPGAS_KEY,

             N_GPUS_KEY,
             GPU_MEMORY_GiB_KEY,
             GPU_P2P_KEY,

             STORAGE_KEY,
             STORAGE_GB_KEY,
             EBS_BANDWIDTH_Gbps_KEY,

             NETWORK_Gbps_KEY))
