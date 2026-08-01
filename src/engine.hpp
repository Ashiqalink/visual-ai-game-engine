#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vector>

namespace vision_engine {

struct Block {
    float x;
    float y;
    float width;
    float height;
    float health;
    float max_health;
    bool active;
};

struct Debris {
    float x;
    float y;
    float vx;
    float vy;
    float width;
    float height;
    float lifespan;
    bool active;
};

class GameEngine {
public:
    GameEngine(float width = 800.0f, float height = 600.0f);

    void update(float dt);
    void set_target_position(float x, float y);
    
    void add_block(float x, float y, float w, float h, float health);
    const std::vector<Block>& get_blocks() const { return m_blocks; }
    const std::vector<Debris>& get_debris() const { return m_debris; }
    void clear_blocks();

    float get_x() const { return m_x; }
    float get_y() const { return m_y; }
    float get_target_x() const { return m_target_x; }
    float get_target_y() const { return m_target_y; }
    float get_width() const { return m_width; }
    float get_height() const { return m_height; }

private:
    float m_width;
    float m_height;
    
    // Ball / Sprite position & physics
    float m_x;
    float m_y;
    float m_vx;
    float m_vy;
    float m_gravity;
    float m_radius;

    // AI Vision Target (e.g. detected face position)
    float m_target_x;
    float m_target_y;

    std::vector<Block> m_blocks;
    std::vector<Debris> m_debris;
};

} // namespace vision_engine

#endif // ENGINE_HPP
