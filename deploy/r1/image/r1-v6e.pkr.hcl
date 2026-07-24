packer {
  required_version = "= 1.15.4"
  required_plugins {
    googlecompute = {
      source  = "github.com/hashicorp/googlecompute"
      version = "= 1.2.7"
    }
  }
}

variable "project_id" {
  type = string
}

variable "build_zone" {
  type = string
}

variable "base_image_self_link" {
  type = string

  validation {
    condition = can(regex(
      "^https://www\\.googleapis\\.com/compute/v1/projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v[0-9]{8}$",
      var.base_image_self_link,
    ))
    error_message = "Base_image_self_link must be one exact dated v6e Ubuntu 22.04 image selfLink."
  }
}

variable "base_image_id" {
  type = number

  validation {
    condition     = var.base_image_id >= 1 && var.base_image_id == floor(var.base_image_id)
    error_message = "Base_image_id must be a positive integer resolved with the selfLink."
  }
}

variable "output_image_name" {
  type = string
}

variable "runtime_archive" {
  type = string
}

variable "runtime_archive_sha256" {
  type = string
}

variable "image_recipe" {
  type = string
}

variable "image_recipe_sha256" {
  type = string
}

variable "provisioner_sha256" {
  type = string
}

locals {
  base_image_segments   = split("/", var.base_image_self_link)
  base_image_project_id = local.base_image_segments[6]
  base_image_name       = local.base_image_segments[9]
}

source "googlecompute" "r1_v6e" {
  project_id              = var.project_id
  zone                    = var.build_zone
  source_image            = local.base_image_name
  source_image_project_id = [local.base_image_project_id]
  image_name              = var.output_image_name
  image_description       = "Unqualified immutable moe-sae DeepSeek-R1 v6e runtime"
  machine_type            = "n2-standard-8"
  disk_size               = 100
  disk_type               = "pd-ssd"
  ssh_username            = "packer"
  use_internal_ip         = true
  # The project has no Cloud NAT. The short-lived baker needs egress for the
  # exact uv, Git, and locked PyPI inputs; Packer still connects over the
  # internal address and deletes the VM after imaging.
  omit_external_ip        = false

  image_labels = {
    workload = "moe-sae-r1"
    state    = "unqualified"
  }
}

build {
  sources = ["source.googlecompute.r1_v6e"]

  provisioner "file" {
    source      = var.runtime_archive
    destination = "/tmp/moe-sae-r1-runtime.tar"
  }

  provisioner "file" {
    source      = var.image_recipe
    destination = "/tmp/moe-sae-r1-image-recipe.json"
  }

  provisioner "shell" {
    execute_command = "/bin/chmod 0700 '{{ .Path }}'; /usr/bin/sudo /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin {{ .Vars }} /bin/bash '{{ .Path }}'"
    environment_vars = [
      "BUILD_TIME_ONLY=1",
      "BASE_IMAGE_ID=${var.base_image_id}",
      "BASE_IMAGE_SELF_LINK=${var.base_image_self_link}",
      "RUNTIME_ARCHIVE_SHA256=${var.runtime_archive_sha256}",
      "IMAGE_RECIPE_SHA256=${var.image_recipe_sha256}",
      "PROVISIONER_SHA256=${var.provisioner_sha256}",
    ]
    script = "${path.root}/provision.sh"
  }
}
