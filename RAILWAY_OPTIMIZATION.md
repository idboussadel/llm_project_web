# Railway Build Optimization

## Problem
Builds were timing out after 11+ minutes due to:
- Slow PyTorch installation
- Large model files being copied
- No build caching

## Solution: Dockerfile with Multi-Stage Build

### Benefits:
1. **Better Caching**: PyTorch installed in separate layer, cached between builds
2. **Faster Installs**: CPU-only PyTorch is smaller and faster
3. **Smaller Image**: Multi-stage build reduces final image size
4. **Build Timeout**: Should complete in 5-7 minutes instead of 11+

### Changes Made:

1. **Dockerfile**: Multi-stage build with optimized layer caching
2. **requirements-optimized.txt**: PyTorch removed (installed separately)
3. **railway.json**: Switched from NIXPACKS to DOCKERFILE builder
4. **.dockerignore**: Excludes unnecessary files from build context

### Build Process:

1. **Builder Stage**: Installs all dependencies
   - PyTorch CPU version (faster, smaller)
   - All other packages
   
2. **Runtime Stage**: Minimal image
   - Only runtime libraries
   - Copied virtual environment
   - Application code

### If Build Still Times Out:

1. **Use Railway's Build Cache**:
   - Railway caches Docker layers automatically
   - First build will be slow, subsequent builds faster

2. **Consider Model Storage**:
   - Store models in S3/Cloud Storage
   - Download on first run instead of bundling
   - Reduces build size significantly

3. **Increase Build Timeout** (Railway Pro):
   - Railway Pro allows longer build times
   - Or use Railway's build settings

4. **Use Pre-built Base Image**:
   - Create a base image with PyTorch pre-installed
   - Push to Docker Hub
   - Use as base in Dockerfile

### Alternative: Split Build

If Dockerfile still times out, consider:
- Building locally and pushing to Railway
- Using Railway's GitHub Actions integration
- Using external CI/CD (GitHub Actions) to build and push

