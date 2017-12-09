# Run AWA with Multiple Workers on Multiple Nodes
This example shows a configuration to distribute the evaluation of objective functions on multiple nodes.

The following configuration runs multiple workers that forward evaluations to different nodes via SSH.
```json
{
  "workers": [
    { "command": "ssh 192.168.0.1  \"./sphere_2D.sh $parameters\"" },
    { "command": "ssh 192.168.0.1  \"./sphere_2D.sh $parameters\"" },
    { "command": "ssh 192.168.0.2  \"./sphere_2D.sh $parameters\"" },
    { "command": "ssh 192.168.0.2  \"./sphere_2D.sh $parameters\"" }
  ]
}
```
To execute commands on remote hosts, users can choose any method.
As well as SSH described above, REST API and RPC are good options to do that.

In any case, users have to setup environments so that workers can execute commands.
The above example needs the following preparation:
- To skip the dialog authentication of SSH log-in, deploy an SSH public key to `192.168.0.1` and `192.168.0.2`, or run `ssh-agent` on a local host.
- Copy `sphere_2D.sh` to the home directory of `192.168.0.1` and `192.168.0.2`.
- Change the permission of `sphere_2D.sh` to be executable in each remote host.

## Run a Demo
This sample script deploys an SSH public key and an objective function program to remote hosts, and then runs AWA.
The script cannot be executed unless you modify remote host settings.
Change `192.168.0.1` and `192.168.0.2` in `config.json` and `run.sh` to your own remote hosts.

After editing, run the script as follows:
```
./run.sh
```
