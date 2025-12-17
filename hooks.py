import logging
import os
import shutil

log = logging.getLogger("mkdocs")


def on_post_build(config, **kwargs):
    """
    Copy the 'web' directory to 'site/web' after the build is complete.
    This ensures the web application is available at /web/ on the deployed site.
    """
    site_dir = config["site_dir"]
    web_source = os.path.join(os.getcwd(), "web")
    web_dest = os.path.join(site_dir, "web")

    if os.path.exists(web_source):
        log.info(f"Copying web app from {web_source} to {web_dest}")
        if os.path.exists(web_dest):
            shutil.rmtree(web_dest)
        shutil.copytree(web_source, web_dest)
    else:
        log.warning(f"Web app source directory {web_source} not found. Skipping copy.")
