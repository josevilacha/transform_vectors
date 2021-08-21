from manim import *

import numpy as np

import itertools

# config.background_color = "#363538"
config.background_color = "#2E3047"
myTemplate = TexTemplate()
myTemplate.add_to_preamble(
    r"\usepackage{fourier} \usepackage[T1]{fontenc} \usepackage{bm}"
)
config.tex_template = myTemplate
config.text_color = "#F6F6F6"
# config.pixel_height = 960  # 1920
# config.pixel_width = 540  # 1080
# config.pixel_height = 1080  #
# config.pixel_width = 1920  #
# config.frame_width = 8
BLUE = "#408697"
ORANGE = "#F58F7C"
RED = "#E9322E"
plate_color = "#707793"

std_line_buff = 0.5
std_wait_time = 3


class VectorScene(Scene):
    def construct(self):

        frame_lin_trans = Rectangle(height=5.0, width=5.0)
        frame_lin_trans.to_edge(LEFT, buff=1)
        self.add(frame_lin_trans)

        frame_text = Rectangle(height=6.0, width=6.0)
        frame_text.to_edge(RIGHT, buff=1)
        # self.add(frame_text)

        title = Tex("Ordinary vectors").scale(1.5)

        all_text_1st_phase = VGroup()

        problem_vector = MathTex(r"{{D}} {{\bm x}} {{=}} {{\bm b}}")
        test = MathTex(r"\sum_i \langle \bm x, \bm v_i\rangle \bm v_i")
        problem_vector_replaced_1_a = MathTex(
            r"{{D}}\sum_i \langle {{\bm x}}, \bm v_i\rangle \bm v_i {{=}} \sum_i \langle {{\bm b}}, \bm v_i\rangle \bm v_i"
        )
        problem_vector_replaced_1_b = MathTex(
            r"{{D}}{{\sum_i}} {{\langle \bm x, \bm v_i\rangle}} {{\bm v_i}} {{=}} {{\sum_i \langle \bm b, \bm v_i\rangle \bm v_i}}"
        )
        problem_vector_replaced_2_a = MathTex(
            r"{{\sum_i}} {{\langle \bm x, \bm v_i\rangle}} {{D}}{{\bm v_i}} {{=}} {{\sum_i \langle \bm b, \bm v_i\rangle \bm v_i}}"
        )
        problem_vector_replaced_2_b = MathTex(
            r"{{\sum_i}} {{\langle \bm x, \bm v_i\rangle}} D{{\bm v_i}} {{=}} {{\sum_i \langle \bm b, \bm v_i\rangle \bm v_i}}"
        )
        problem_vector_replaced_3_a = MathTex(
            r"{{\sum_i}} {{\langle \bm x, \bm v_i\rangle}}\lambda_i{{\bm v_i}} {{=}} {{\sum_i \langle \bm b, \bm v_i\rangle \bm v_i}}"
        )
        problem_vector_replaced_3_b = MathTex(
            r"\sum_i {{\langle \bm x, \bm v_i\rangle\lambda_i}}\bm v_i {{=}} \sum_i {{\langle \bm b, \bm v_i\rangle}} \bm v_i"
        )
        problem_vector_replaced_4_a = MathTex(
            r"{{\langle \bm x, \bm v_i\rangle\lambda_i}} {{=}} {{\langle \bm b, \bm v_i\rangle}},\quad i=1,2"
        )
        problem_vector_replaced_4_b = MathTex(
            r"{{\langle \bm x, \bm v_i\rangle}}{{\lambda_i}} {{=}} {{\langle \bm b, \bm v_i\rangle}}{{,\quad i=1,2}}"
        )
        problem_vector_replaced_5 = MathTex(
            r"{{\langle \bm x, \bm v_i\rangle}} {{=}} {{\langle \bm b, \bm v_i\rangle}}/{{\lambda_i}}{{,\quad i=1,2}}"
        )
        all_text_1st_phase.add(problem_vector)

        text_lin_trans = Tex("$D$ is a linear tranformation")
        all_text_1st_phase.add(text_lin_trans)

        question_eigen = Tex("Ideally, what is the simplest effect $D$ could have?")
        eigen_problem_vector = MathTex(
            r"{{D \bm v = \lambda \bm v}} \to \begin{cases} \bm v_i,\\ \lambda_i, \end{cases} i=1,2"
        )
        all_text_1st_phase.add(eigen_problem_vector)

        vec_in_phase_space = MathTex(
            r"\bm x = \sum_i {{\langle \bm x, \bm v_i\rangle}} \bm v_i"
        )
        all_text_1st_phase.add(vec_in_phase_space)

        title.move_to(frame_text.get_center())
        title.to_edge(UP, buff=1)
        self.add(title)
        all_text_1st_phase.arrange(DOWN, buff=std_line_buff)
        all_text_1st_phase.move_to(frame_text.get_center())
        all_text_1st_phase.next_to(title, DOWN, buff=0.8)
        self.add(all_text_1st_phase)

        problem_vector_replaced_1_a.move_to(problem_vector.get_center())
        test.move_to(problem_vector.get_center())
        problem_vector_replaced_1_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_b.move_to(problem_vector.get_center())
        problem_vector_replaced_4_a.move_to(problem_vector.get_center())
        problem_vector_replaced_4_b.move_to(problem_vector.get_center())
        problem_vector_replaced_5.move_to(problem_vector.get_center())
        self.wait(2)
        self.play(Indicate(vec_in_phase_space, color=ORANGE))
        self.wait(1)
        self.play(
            # ReplacementTransform(problem_vector.get_part_by_tex(r"\bm x"), test),
            TransformMatchingTex(problem_vector, problem_vector_replaced_1_a),
        )
        self.wait(2)
        self.remove(problem_vector_replaced_1_a)
        self.add(problem_vector_replaced_1_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_1_b,
                problem_vector_replaced_2_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_2_a)
        self.add(problem_vector_replaced_2_b)
        self.play(
            Indicate(
                eigen_problem_vector.get_part_by_tex(r"D \bm v = \lambda \bm v"),
                color=ORANGE,
            ),
        )
        self.wait(1)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_b,
                problem_vector_replaced_3_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_3_a)
        self.add(problem_vector_replaced_3_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_3_b,
                problem_vector_replaced_4_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_4_a)
        self.add(problem_vector_replaced_4_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_4_b,
                problem_vector_replaced_5,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.play(
            Indicate(
                problem_vector_replaced_5.get_part_by_tex(
                    r"\langle \bm x, \bm v_i\rangle"
                ),
                color=ORANGE,
            ),
            Indicate(
                vec_in_phase_space.get_part_by_tex(r"\langle \bm x, \bm v_i\rangle"),
                color=ORANGE,
            ),
        )

        self.wait(10)


class Matrix(LinearTransformationScene):
    def __init__(self):
        global config
        # config.pixel_height = 1080  #
        # config.pixel_width = 1080  #
        config.frame_height = 5
        config.frame_width = 5

        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=True,
            show_basis_vectors=True,
        )

    def construct(self):

        matrix = [[1, 2], [2, 1]]

        matrix_tex = (
            MathTex("A=\\begin{bmatrix} 1 & 2\\\ 2 & 1 \\end{bmatrix}")
            .to_edge(UL)
            .add_background_rectangle()
        )

        unit_square = self.get_unit_square()
        text = always_redraw(
            lambda: MathTex("\\operatorname{Det}(A)")
            .set(width=0.7)
            .move_to(unit_square.get_center())
        )

        vect = self.get_vector([1, -2], color=PURPLE_B)

        rect1 = Rectangle(
            height=2, width=1, stroke_color=BLUE_A, fill_color=BLUE_D, fill_opacity=0.6
        ).shift(UP * 2 + LEFT * 2)

        circ1 = Circle(
            radius=1, stroke_color=BLUE_A, fill_color=BLUE_D, fill_opacity=0.6
        ).shift(DOWN * 2 + RIGHT * 1)

        self.add_transformable_mobject(vect, unit_square, rect1, circ1)
        self.add_background_mobject(matrix_tex, text)
        self.apply_matrix(matrix)

        self.wait()


class Move(Scene):
    def __init__(self):
        config.pixel_height = 1080  #
        config.pixel_width = 1080  #
        config.frame_height = 5
        config.frame_width = 5
        Scene.__init__(self)

    def construct(self):

        # dot = Dot(ORIGIN)
        # arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
        # numberplane = NumberPlane()
        # origin_text = Text("(0, 0)").next_to(dot, DOWN)
        # tip_text = Text("(2, 2)").next_to(arrow.get_end(), RIGHT)
        # self.add(numberplane, dot, arrow, origin_text, tip_text)

        lin_trans = np.array([[3, 0.3], [0.3, 4]]) / 1.8
        plane = NumberPlane(
            x_range=[-5, 5, 0.25], y_range=[-4, 4, 0.25], x_length=10, y_length=10
        )

        eig_vals, eig_vecs = np.linalg.eig(lin_trans)

        eig_vec_1 = Arrow(
            ORIGIN,
            [comp for comp in eig_vecs[:, 0]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=ORANGE,
        )
        eig_vec_1_to_transform = eig_vec_1.copy()
        eig_vec_1_transformed = Arrow(
            ORIGIN,
            [comp * eig_vals[0] for comp in eig_vecs[:, 0]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=GREEN,
        )

        eig_vec_2 = Arrow(
            ORIGIN,
            [comp for comp in eig_vecs[:, 1]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=ORANGE,
        )
        eig_vec_2_to_transform = eig_vec_2.copy()
        eig_vec_2_transformed = Arrow(
            ORIGIN,
            [comp * eig_vals[1] for comp in eig_vecs[:, 1]] + [0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=GREEN,
        )

        self.play(DrawBorderThenFill(plane), run_time=2)
        self.add(eig_vec_1)
        self.add(eig_vec_2)

        self.play(
            plane.animate.apply_matrix(lin_trans),
            eig_vec_1_to_transform.animate.apply_matrix(lin_trans),
            eig_vec_2_to_transform.animate.apply_matrix(lin_trans),
        )
        self.remove(eig_vec_1_to_transform)
        self.add(eig_vec_1_transformed)
        self.add(eig_vec_1)
        self.remove(eig_vec_2_to_transform)
        self.add(eig_vec_2_transformed)
        self.add(eig_vec_2)

        self.wait(2)

        comp_1_tex = MathTex(r"\langle \bm x, \bm v_1\rangle")
        comp_2_tex = MathTex(r"\langle \bm x, \bm v_2\rangle")
        random_vec_coord = np.array([1, 1, 0])
        random_vec = Arrow(
            ORIGIN,
            random_vec_coord,
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=10,
            max_stroke_width_to_length_ratio=20,
            color=BLUE,
        )
        eig_1 = np.zeros(3)
        eig_2 = np.zeros(3)
        eig_1[:2] = eig_vecs[:, 0]
        eig_2[:2] = eig_vecs[:, 1]
        comp_1 = eig_1.dot(random_vec_coord)
        comp_2 = eig_2.dot(random_vec_coord)
        brace_comp_1 = BraceBetweenPoints(ORIGIN, comp_1 * eig_1)
        line_comp_1 = Line(random_vec_coord, random_vec_coord - comp_1 * eig_1)
        line_comp_1.stroke_width = 1
        line_comp_1.opacity = 0.5
        line_comp_2 = Line(random_vec_coord, random_vec_coord - comp_2 * eig_2)
        brace_comp_2 = BraceBetweenPoints(comp_2 * eig_2, ORIGIN)
        line_comp_2.stroke_width = 1
        line_comp_2.opacity = 0.5
        self.add(line_comp_1)
        self.add(line_comp_2)
        self.play(
            *[
                DrawBorderThenFill(mobject)
                for mobject in [
                    random_vec,
                    line_comp_1,
                    line_comp_2,
                    brace_comp_1,
                    brace_comp_2,
                ]
            ],
            run_time=std_wait_time
        )
        comp_1_tex.next_to(brace_comp_2, LEFT)
        comp_2_tex.next_to(brace_comp_1, DOWN)
        # comp_1_tex.add_background_rectangle()
        # comp_2_tex.add_background_rectangle()
        self.play(*[Write(mobject) for mobject in [comp_1_tex, comp_2_tex]])

        self.wait(10)

        # f_tex = r"f(x)"
        # equation = MathTex("dA", "\\approx", f_tex, "dx")
        # # equation.to_edge(RIGHT).shift(3*UP)
        # deriv_equation = MathTex("{dA", "\\over \\,", "dx}", "\\approx", f_tex)
        # deriv_equation.move_to(equation, UP + LEFT)
        #
        # self.play(
        #     *[
        #         ReplacementTransform(
        #             equation.get_part_by_tex(tex),
        #             deriv_equation.get_part_by_tex(tex),
        #             run_time=10,
        #         )
        #         for tex in ("dA", "approx", f_tex, "dx")
        #     ]
        #     + [Write(deriv_equation.get_part_by_tex("over"))]
        # )
