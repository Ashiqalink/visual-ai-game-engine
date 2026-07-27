#include "engine.hpp"
#include <cmath>
#include <algorithm>

namespace vision_engine {

GameEngine::GameEngine(float width, float height)
    : m_width(width), m_height(height),
      m_x(width / 2.0f), m_y(height / 4.0f),
      m_vx(120.0f), m_vy(0.0f),
      m_gravity(400.0f), m_radius(25.0f),
      m_target_x(width / 2.0f), m_target_y(height / 2.0f) {}

void GameEngine::set_target_position(float x, float y) {
    m_target_x = x;
    m_target_y = y;
}

void GameEngine::update(float dt) {
    // Apply gravity
    m_vy += m_gravity * dt;

    // Attraction force towards vision target (e.g., face position)
    float dx = m_target_x - m_x;
    float dy = m_target_y - m_y;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist > 1.0f) {
        float pull_strength = 150.0f;
        m_vx += (dx / dist) * pull_strength * dt;
        m_vy += (dy / dist) * pull_strength * dt;
    }

    // Integrate position
    m_x += m_vx * dt;
    m_y += m_vy * dt;

    // Damping / Friction
    m_vx *= 0.99f;

    // Boundary collisions
    if (m_x - m_radius < 0.0f) {
        m_x = m_radius;
        m_vx = -m_vx * 0.8f;
    } else if (m_x + m_radius > m_width) {
        m_x = m_width - m_radius;
        m_vx = -m_vx * 0.8f;
    }

    if (m_y - m_radius < 0.0f) {
        m_y = m_radius;
        m_vy = -m_vy * 0.8f;
    } else if (m_y + m_radius > m_height) {
        m_y = m_height - m_radius;
        m_vy = -m_vy * 0.8f;
    }
}

} // namespace vision_engine
