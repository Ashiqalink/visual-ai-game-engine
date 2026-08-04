#include "engine.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>

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

int GameEngine::add_entity(std::string name, float x, float y, float z,
                           float vx, float vy, float vz,
                           float w, float h, float d,
                           Material mat) {
    int id = m_next_entity_id++;
    m_entities.push_back({id, name, x, y, z, vx, vy, vz, w, h, d, true, mat});
    return id;
}

void GameEngine::clear_entities() {
    m_entities.clear();
}

void GameEngine::add_block(float x, float y, float w, float h, float health, Material mat) {
    m_blocks.push_back({x, y, w, h, health, health, true, mat});
}

void GameEngine::clear_blocks() {
    m_blocks.clear();
    m_debris.clear();
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

    // Update general entities
    for (auto& entity : m_entities) {
        if (!entity.active) continue;
        entity.x += entity.vx * dt;
        entity.y += entity.vy * dt;
        entity.z += entity.vz * dt;
    }

    // Block collisions
    for (auto& block : m_blocks) {
        if (!block.active) continue;

        // Simple AABB vs Circle collision
        // Find closest point on block to circle
        float closestX = std::max(block.x - block.width/2.0f, std::min(m_x, block.x + block.width/2.0f));
        float closestY = std::max(block.y - block.height/2.0f, std::min(m_y, block.y + block.height/2.0f));

        float distObjX = m_x - closestX;
        float distObjY = m_y - closestY;
        float distance = std::sqrt(distObjX * distObjX + distObjY * distObjY);

        if (distance < m_radius) {
            // Collision occurred
            // Determine bounce direction
            if (distance > 0) {
                float nx = distObjX / distance;
                float ny = distObjY / distance;
                // Move out of collision
                m_x = closestX + nx * m_radius;
                m_y = closestY + ny * m_radius;
                
                // Calculate impact speed for damage
                float impact = std::sqrt(m_vx * m_vx + m_vy * m_vy);
                
                // Reflect velocity
                float dotProduct = (m_vx * nx + m_vy * ny);
                if (dotProduct < 0) {
                    m_vx -= 1.6f * dotProduct * nx; // bounce restitution 0.8 * 2 = 1.6
                    m_vy -= 1.6f * dotProduct * ny;
                    
                    // Apply damage
                    if (impact > 50.0f) {
                        block.health -= impact * 0.1f;
                        if (block.health <= 0.0f) {
                            block.active = false;
                            
                            // Spawn debris (2x2 grid)
                            float dw = block.width / 2.0f;
                            float dh = block.height / 2.0f;
                            for (int i = 0; i < 2; ++i) {
                                for (int j = 0; j < 2; ++j) {
                                    float dx_offset = (i == 0) ? -dw/2.0f : dw/2.0f;
                                    float dy_offset = (j == 0) ? -dh/2.0f : dh/2.0f;
                                    float dvx = ((std::rand() % 100) / 50.0f - 1.0f) * 100.0f + m_vx * 0.2f;
                                    float dvy = ((std::rand() % 100) / 50.0f - 1.0f) * 100.0f + m_vy * 0.2f;
                                    m_debris.push_back({
                                        block.x + dx_offset, block.y + dy_offset,
                                        dvx, dvy,
                                        dw, dh,
                                        3.0f, // 3 seconds lifespan
                                        true,
                                        block.material
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Update debris
    for (auto& d : m_debris) {
        if (!d.active) continue;
        d.vy += m_gravity * dt;
        d.x += d.vx * dt;
        d.y += d.vy * dt;
        d.lifespan -= dt;

        // Debris Floor collision
        if (d.y + d.height/2.0f > m_height) {
            d.y = m_height - d.height/2.0f;
            d.vy = -d.vy * 0.5f;
            d.vx *= 0.8f;
        }

        if (d.lifespan <= 0.0f) {
            d.active = false;
        }
    }
}

} // namespace vision_engine
