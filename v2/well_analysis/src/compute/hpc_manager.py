import paramiko
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import subprocess
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class HPCManager:
    def __init__(self, config_path: Union[str, Path] = None):
        """Initialize HPC Manager"""
        self.config = self._load_config(config_path)
        self.ssh_client = None
        self.sftp_client = None

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load HPC configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'hpc_config.json'

        try:
            with open(config_path) as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading HPC config: {str(e)}")
            return {
                "host": "",
                "username": "",
                "key_path": "",
                "work_dir": "",
                "partition": "compute",
                "time": "24:00:00",
                "mem_per_cpu": "4G",
                "cpus_per_task": "4"
            }

    def connect(self) -> bool:
        """Establish SSH connection to HPC cluster"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                hostname=self.config['host'],
                username=self.config['username'],
                key_filename=self.config['key_path']
            )
            self.sftp_client = self.ssh_client.open_sftp()
            return True
        except Exception as e:
            logger.error(f"Error connecting to HPC: {str(e)}")
            return False

    def disconnect(self):
        """Close SSH connection"""
        try:
            if self.sftp_client:
                self.sftp_client.close()
            if self.ssh_client:
                self.ssh_client.close()
        except Exception as e:
            logger.error(f"Error disconnecting from HPC: {str(e)}")

    def generate_job_script(self, job_name: str,
                            script_content: str,
                            additional_params: Dict = None) -> str:
        """Generate SLURM job script"""
        try:
            script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.config['partition']}
#SBATCH --time={self.config['time']}
#SBATCH --mem-per-cpu={self.config['mem_per_cpu']}
#SBATCH --cpus-per-task={self.config['cpus_per_task']}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
"""

            # Add additional SLURM parameters
            if additional_params:
                for key, value in additional_params.items():
                    script += f"#SBATCH --{key}={value}\n"

            # Add module loading
            for module in self.config.get('modules', []):
                script += f"\nmodule load {module}"

            # Add main script content
            script += f"\n\n{script_content}\n"

            return script

        except Exception as e:
            logger.error(f"Error generating job script: {str(e)}")
            return ""

    def submit_job(self, script_content: str,
                   job_name: str,
                   additional_params: Dict = None) -> Optional[str]:
        """Submit job to HPC cluster"""
        try:
            if not self.ssh_client:
                if not self.connect():
                    return None

            # Generate job script
            script = self.generate_job_script(
                job_name, script_content, additional_params)

            # Create work directory if it doesn't exist
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"mkdir -p {self.config['work_dir']}")

            # Write script to remote file
            script_path = f"{self.config['work_dir']}/{job_name}.sh"
            with self.sftp_client.open(script_path, 'w') as f:
                f.write(script)

            # Submit job
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"cd {self.config['work_dir']} && sbatch {job_name}.sh")

            # Get job ID
            job_id = stdout.read().decode().strip().split()[-1]

            return job_id

        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            return None

    def check_job_status(self, job_id: str) -> str:
        """Check status of submitted job"""
        try:
            if not self.ssh_client:
                if not self.connect():
                    return "UNKNOWN"

            cmd = f"sacct -j {job_id} --format=State -n -P"
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
            status = stdout.read().decode().strip()

            return status if status else "UNKNOWN"

        except Exception as e:
            logger.error(f"Error checking job status: {str(e)}")
            return "UNKNOWN"

    def get_job_output(self, job_id: str, job_name: str) -> Dict[str, str]:
        """Retrieve job output files"""
        try:
            if not self.ssh_client:
                if not self.connect():
                    return {}

            output = {}
            for file_type in ['out', 'err']:
                file_path = f"{self.config['work_dir']}/{job_name}_{job_id}.{file_type}"
                try:
                    with self.sftp_client.open(file_path, 'r') as f:
                        output[file_type] = f.read().decode()
                except:
                    output[file_type] = ""

            return output

        except Exception as e:
            logger.error(f"Error getting job output: {str(e)}")
            return {}

    def download_results(self, job_id: str, job_name: str,
                         local_dir: Union[str, Path]) -> bool:
        """Download job results to local directory"""
        try:
            if not self.ssh_client:
                if not self.connect():
                    return False

            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            # Get list of result files
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"ls {self.config['work_dir']}/{job_name}*")
            remote_files = stdout.read().decode().strip().split('\n')

            # Download each file
            for remote_path in remote_files:
                if remote_path:
                    filename = os.path.basename(remote_path)
                    local_path = local_dir / filename
                    self.sftp_client.get(remote_path, str(local_path))

            return True

        except Exception as e:
            logger.error(f"Error downloading results: {str(e)}")
            return False

    def cleanup_job_files(self, job_id: str, job_name: str) -> bool:
        """Clean up job-related files on HPC cluster"""
        try:
            if not self.ssh_client:
                if not self.connect():
                    return False

            # Remove job files
            cmd = f"rm -f {self.config['work_dir']}/{job_name}*"
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)

            return True

        except Exception as e:
            logger.error(f"Error cleaning up job files: {str(e)}")
            return False

    def monitor_job(self, job_id: str, job_name: str,
                    check_interval: int = 60) -> Dict:
        """Monitor job until completion"""
        try:
            status = "PENDING"
            start_time = time.time()

            while status not in ['COMPLETED', 'FAILED', 'CANCELLED']:
                status = self.check_job_status(job_id)

                if status == "RUNNING":
                    # Get resource usage
                    cmd = f"sstat -j {job_id} --format=MaxRSS,MaxVMSize,AveCPU -n -P"
                    stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
                    resources = stdout.read().decode().strip()

                    logger.info(f"Job {job_id} resource usage: {resources}")

                time.sleep(check_interval)

            duration = time.time() - start_time

            # Get job output
            output = self.get_job_output(job_id, job_name)

            return {
                'job_id': job_id,
                'status': status,
                'duration': duration,
                'output': output
            }

        except Exception as e:
            logger.error(f"Error monitoring job: {str(e)}")
            return {}

    def run_analysis(self, analysis_type: str,
                     analysis_params: Dict,
                     job_name: Optional[str] = None) -> Dict:
        """Run analysis job on HPC cluster"""
        try:
            if job_name is None:
                job_name = f"{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create Python script for analysis
            script_content = f"""
import json
import sys
from well_analysis.analysis import {analysis_type}

# Load parameters
params = json.loads('''{json.dumps(analysis_params)}''')

# Run analysis
analyzer = {analysis_type}.Analyzer()
results = analyzer.run_analysis(params)

# Save results
with open('{job_name}_results.json', 'w') as f:
    json.dump(results, f)
"""

            # Submit job
            job_id = self.submit_job(script_content, job_name)
            if not job_id:
                return {}

            # Monitor job
            job_result = self.monitor_job(job_id, job_name)

            # Download results if job completed successfully
            if job_result.get('status') == 'COMPLETED':
                self.download_results(job_id, job_name, 'results')

            # Cleanup
            self.cleanup_job_files(job_id, job_name)

            return job_result

        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return {}
