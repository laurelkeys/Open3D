// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/registration/Registration.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

using namespace open3d::visualization;

const int WIDTH = 1024;
const int HEIGHT = 768;
float verticalFoV = 25;

const Eigen::Vector3f CENTER_OFFSET(-10.0f, 0.0f, 30.0f);

const std::string SRC_CLOUD = "source_pointcloud";
const std::string DST_CLOUD = "target_pointcloud";
const std::string LINE_SET = "correspondences_lines";
const std::string SRC_CORRES = "source_correspondences_idx";
const std::string TAR_CORRES = "target_correspondences_idx";

class ReconstructionWindow : public gui::Window {
    using Super = gui::Window;

public:
    ReconstructionWindow() : gui::Window(window_name, WIDTH, HEIGHT) {
        auto& theme = GetTheme();
        int em = theme.font_size;
        int spacing = int(std::round(0.5f * float(em)));
        gui::Margins margins(int(std::round(0.5f * float(em))));

        widget3d_ = std::make_shared<gui::SceneWidget>();
        output_panel_ = std::make_shared<gui::Vert>(spacing, margins);

        AddChild(widget3d_);
        AddChild(output_panel_);

        output_ = std::make_shared<gui::Label>("");
        const char* label = widget_string.c_str();
        output_panel_->AddChild(std::make_shared<gui::Label>(label));
        output_panel_->AddChild(output_);

        widget3d_->SetScene(
                std::make_shared<rendering::Open3DScene>(GetRenderer()));
    }

    ~ReconstructionWindow() {}

    void Layout(const gui::LayoutContext& context) override {
        int em = context.theme.font_size;
        int panel_width = 15 * em;
        int panel_height = 100;
        // The usable part of the window may not be the full size if there
        // is a menu.
        auto content_rect = GetContentRect();

        output_panel_->SetFrame(gui::Rect(content_rect.GetRight() - panel_width,
                                          content_rect.y, panel_width,
                                          panel_height));
        int x = content_rect.x;
        widget3d_->SetFrame(gui::Rect(x, content_rect.y, content_rect.width,
                                      content_rect.height));
        Super::Layout(context);
    }

protected:
    std::shared_ptr<gui::Vert> output_panel_;
    std::shared_ptr<gui::Label> output_;
    std::shared_ptr<gui::SceneWidget> widget3d_;

    void SetOutput(const std::string& output) {
        output_->SetText(output.c_str());
    }
};

class MultiScaleICPVisualizer : public ReconstructionWindow {
public:
    MultiScaleICPVisualizer(const geometry::PointCloud source_pcd,
                            const geometry::PointCloud target_pcd,
                            const core::Tensor initial_transformation,
                            const core::Device& device)
        : device_(source_pcd.GetDevice()),
          host_(core::Device("CPU:0")),
          dtype_(source_pcd.GetDtype()) {
        ReadConfigFile(path_config);

        sleep_time_ = 100;

        // When window is closed, it will stop the execute of the code.
        is_done_ = false;
        SetOnClose([this]() {
            is_done_ = true;
            return true;  // false would cancel the close
        });
        update_thread_ = std::thread([&]() {
            this->UpdateMain(source_pcd, target_pcd, initial_transformation);
        });
    }

    ~MultiScaleICPVisualizer() { update_thread_.join(); }

public:
    void Update(const core::Tensor& delta_transformation,
                const geometry::PointCloud& correspondences_source_pcd,
                const geometry::PointCloud& correspondences_target_pcd) {
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            correspondence_src_pcd = correspondences_source_pcd.To(host_);
            correspondence_tar_pcd = correspondences_target_pcd.To(host_);

            src_pcd_.Transform(delta_transformation);
        }

        gui::Application::GetInstance().PostToMainThread(this, [&]() {
            std::lock_guard<std::mutex> lock(cloud_lock_);

            this->widget3d_->GetScene()->GetScene()->UpdateGeometry(
                    SRC_CLOUD, src_pcd_,
                    rendering::Scene::kUpdatePointsFlag |
                            rendering::Scene::kUpdateColorsFlag);
            this->widget3d_->GetScene()->GetScene()->UpdateGeometry(
                    SRC_CORRES, correspondence_src_pcd,
                    rendering::Scene::kUpdatePointsFlag |
                            rendering::Scene::kUpdateColorsFlag);
            this->widget3d_->GetScene()->GetScene()->UpdateGeometry(
                    TAR_CORRES, correspondence_tar_pcd,
                    rendering::Scene::kUpdatePointsFlag |
                            rendering::Scene::kUpdateColorsFlag);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }

    void ClearCorrespondences() {
        gui::Application::GetInstance().PostToMainThread(this, [this]() {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            this->widget3d_->GetScene()->GetScene()->RemoveGeometry(SRC_CORRES);
            this->widget3d_->GetScene()->GetScene()->RemoveGeometry(TAR_CORRES);
        });
    }

private:
    std::thread update_thread_;
    std::mutex cloud_lock_;
    int sleep_time_;

    void UpdateMain(geometry::PointCloud& source_pcd,
                    geometry::PointCloud& target_pcd,
                    core::Tensor& initial_transformation) {
        {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            src_pcd_ = source_pcd.Transform(dtype_, device_).CPU();
            tar_pcd_ = target_pcd.CPU();
        }

        gui::Application::GetInstance().PostToMainThread(this, [this]() {
            std::lock_guard<std::mutex> lock(cloud_lock_);
            this->widget3d_->GetScene()->SetBackground({0, 0, 0, 1});

            this->widget3d_->GetScene()->AddGeometry(DST_CLOUD, &tar_pcd_,
                                                     tar_cloud_mat_);

            this->widget3d_->GetScene()->GetScene()->AddGeometry(
                    SRC_CLOUD, src_pcd_, src_cloud_mat_);
            this->widget3d_->GetScene()->GetScene()->AddGeometry(
                    SRC_CORRES, src_pcd_, src_corres_mat_);
            this->widget3d_->GetScene()->GetScene()->AddGeometry(
                    TAR_CORRES, src_pcd_, tar_corres_mat_);

            auto bbox = this->widget3d_->GetScene()->GetBoundingBox();
            auto center = bbox.GetCenter().cast<float>();
            this->widget3d_->SetupCamera(18, bbox, center);
            this->widget3d_->LookAt(center, center - Eigen::Vector3f{-10, 5, 8},
                                    {0.0f, -1.0f, 0.0f});
        });
    }

private:
    std::mutex cloud_lock_;

    std::atomic<bool> is_done_;
    open3d::visualization::rendering::Material src_cloud_mat_;
    open3d::visualization::rendering::Material tar_cloud_mat_;
    open3d::visualization::rendering::Material src_corres_mat_;
    open3d::visualization::rendering::Material tar_corres_mat_;

    // For Visualization.
    t::geometry::PointCloud correspondence_src_pcd;
    t::geometry::PointCloud correspondence_tar_pcd;
    t::geometry::PointCloud src_pcd_;
    t::geometry::PointCloud tar_pcd_;

    t::geometry::PointCloud source_;
    t::geometry::PointCloud target_;

    core::Tensor correspondences_source_idx_;
    core::Tensor correspondences_target_idx_;

    core::Tensor transformation_;
    RegistrationResult result_;

    core::Device device_;
    core::Dtype dtype_;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
