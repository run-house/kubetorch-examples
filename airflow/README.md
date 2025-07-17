# Kubetorch + Airflow

This is a standalone example that demonstrates a simple use of Kubetorch inside an Airflow DAG.

## Installation & Setup

Ideally, you should install Kubetorch and Airflow in the same Kubernetes cluster. This will simplify connections between services and improve overall reliability.

To illustrate how to set up Airflow + Kubetorch on a fresh cluster, we'll walk through deploying Airflow with Helm, using a Docker image with Airflow Kubetorch Python package and our training DAG.

You will need a Kubernetes cluster, a local Kubeconfig, and a place to store Docker images.

1. Replace `<KT_VERSION>`, `KT_LICENSE_KEY`, and `<API_KEY>` in the URL in `docker/requirements.txt`.
   You can also directly use the URL provided in our [Onboarding Guide](https://www.run.house/kubetorch/get-started)
2. Build the Docker image, tagging an appropriate location in your registry (e.g. ECR, GCR). This should be accessible from your Kubernetes cluster.
   ```bash
   docker build --platform linux/amd64 -t your-registry/kt-airflow-custom:latest -f docker/Dockerfile .
   ```
3. Push the image to your container registry.
   ```bash
   docker push your-registry/kt-airflow-custom:latest
   ```
4. Use Helm to install Airflow with our custom image.
   ```bash
   helm repo add apache-airflow https://airflow.apache.org
   helm repo update
   kubectl create namespace airflow
   helm install airflow apache-airflow/airflow \
     --namespace airflow \
     --set images.airflow.repository=your-registry/kt-airflow-custom \
     --set images.airflow.tag=latest
   ```
   Depending on your cluster setup, may need to use a custom `k8s/values.yaml` file for the installation. The one in this folder works with the Kubetorch Terraform install on EKS.
5. Apply RBAC to the Airflow worker service account to allow for Kubetorch dispatching operations.
   ```bash
   kubectl apply -f k8s/rbac.yaml
   ```

### Connecting to Airflow

```bash
kubectl port-forward svc/airflow-api-server 8080:8080 --namespace airflow
```

Then navigate to http://localhost:8080 in your browser to view the Airflow UI.

By default, you should be able to log in with username `admin` and password `admin`.

### Monitoring and Debugging

The following Kubernetes commands will be useful for debugging:

```bash
# List all airflow pods
kubectl get pods -n [your-namespace] | grep airflow

# View logs for specific components
kubectl logs -f deployment/airflow-scheduler -n [your-namespace]
kubectl logs -f deployment/airflow-webserver -n [your-namespace]
kubectl logs -f deployment/airflow-worker -n [your-namespace]
```

To see logs from the Kubetorch methods, launching compute and more, you'll likely be most interested in `worker`.
