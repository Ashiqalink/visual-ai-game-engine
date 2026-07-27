#ifndef ENGINE_HPP
#define ENGINE_HPP

namespace vision_engine {

class GameEngine {
public:
    GameEngine(float width = 800.0f, float height = 600.0f);

    void update(float dt);
    void set_target_position(float x, float y);
    
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
};

} // namespace vision_engine

#endif // ENGINE_HPP
