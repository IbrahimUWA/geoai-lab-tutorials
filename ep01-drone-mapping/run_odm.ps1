# GeoAI Lab - Episode 1: run OpenDroneMap on a folder of drone photos (Windows PowerShell)
# Layout expected:   <data>\aukerman\images\*.JPG
# Outputs land in:   <data>\aukerman\odm_orthophoto, odm_dem, odm_georeferencing, odm_report
param(
    [string]$Data = "E:\GeoAI-Lab-Tutorials\data\drone",
    [string]$Project = "aukerman"
)
docker run --rm -v "${Data}:/datasets" opendronemap/odm `
    --project-path /datasets $Project `
    --dsm --dtm `
    --orthophoto-resolution 5 `
    --dem-resolution 10 `
    --pc-quality medium `
    --feature-quality high `
    --max-concurrency 10
