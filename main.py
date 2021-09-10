from manim import *

import numpy as np

import itertools

# config.background_color = "#363538"
config.background_color = "#2E3047"
myTemplate = TexTemplate()
myTemplate.add_to_preamble(
    r"\usepackage{fourier} \usepackage[T1]{fontenc} \usepackage{bm} \usepackage{mathtools}"
)
config.tex_template = myTemplate
config.text_color = "#F6F6F6"
config.pixel_height = 1080  #
config.pixel_width = 1920  #
config.frame_width = 18
BLUE = "#707793"
ORANGE = "#F58F7C"
RED = "#E9322E"
plate_color = "#707793"

std_line_buff = 0.5
std_wait_time = 3


class VectorScene(Scene):
    def construct(self):

        frame_lin_trans = Rectangle(height=6.0, width=6.0)
        frame_lin_trans.to_edge(LEFT, buff=1)
        self.add(frame_lin_trans)

        frame_text = Rectangle(height=6.0, width=7.0)
        frame_text.to_edge(RIGHT, buff=1)
        frame_u = Rectangle(height=2, width=14.0)
        large_v_group = VGroup(frame_lin_trans, frame_text)
        large_v_group.arrange(RIGHT, buff=1)
        full_group = VGroup(frame_u, large_v_group)
        full_group.arrange(DOWN)
        full_group.center()

        # self.add(full_group)

        # self.add(frame_text)

        title = Tex("Ordinary vectors").scale(1.5)

        all_text_1st_phase = VGroup()

        diff_eq = MathTex(r"mD^2 f(t) + c {{Df(t)}} + kf(t) {{=}} F(t)", color=BLUE)
        diff_eq_simple = MathTex(r"{{Df(t)}}{{=}} {{F(t)}}", color=BLUE)

        problem_vector = MathTex(r"{{D\bm x}} {{=}} {{\bm b}}", color=BLUE)
        problem_vector_replaced_1_a = MathTex(
            r"\sum_i ( {{D\bm x}}, \bm v_i) \bm v_i {{=}} \sum_i ( {{\bm b}}, \bm v_i) \bm v_i",
            color=BLUE,
        )
        problem_vector_replaced_1_b = MathTex(
            r"{{\sum_i (}} {{D}}{{\bm x}}, {{\bm v_i)}} {{\bm v_i}} {{=}} {{\sum_i ( \bm b, \bm v_i) \bm v_i}}",
            color=BLUE,
        )
        problem_vector_replaced_2_a = MathTex(
            r"{{\sum_i (}}{{\bm x}}, {{D}}^T{{\bm v_i)}} {{\bm v_i}} {{=}} {{\sum_i ( \bm b, \bm v_i) \bm v_i}}",
            color=BLUE,
        )
        problem_vector_replaced_2_b = MathTex(
            r"{{\sum_i (\bm x}}, {{D}}^T{{\bm v_i) \bm v_i = \sum_i ( \bm b, \bm v_i) \bm v_i}}",
            color=BLUE,
        )
        problem_vector_replaced_2_5_a = MathTex(
            r"{{\sum_i (\bm x}}, {{D}}{{\bm v_i) \bm v_i = \sum_i ( \bm b, \bm v_i) \bm v_i}}",
            color=BLUE,
        )
        problem_vector_replaced_2_5_b = MathTex(
            r"{{\sum_i}} {{( \bm x, D\bm v_i)}} {{\bm v_i}} {{=}} {{\sum_i ( \bm b, \bm v_i) \bm v_i}}",
            color=BLUE,
        )
        problem_vector_replaced_3_a = MathTex(
            r"{{\sum_i}} {{( \bm x, \lambda_i\bm v_i)}}{{\bm v_i}} {{=}} {{\sum_i ( \bm b, \bm v_i) \bm v_i}}",
            color=BLUE,
        )
        problem_vector_replaced_3_b = MathTex(
            r"{{\sum_i}} {{( \bm x,}}{{\lambda_i}} {{\bm v_i)\bm v_i}} {{=}} \sum_i {{( \bm b, \bm v_i)}} \bm v_i",
            color=BLUE,
        )
        problem_vector_replaced_3_5_a = MathTex(
            r"{\sum_i} {{\lambda_i}}{{( \bm x,}} {{\bm v_i)\bm v_i}} {{=}} \sum_i {{( \bm b, \bm v_i)}} \bm v_i",
            color=BLUE,
        )
        problem_vector_replaced_3_5_b = MathTex(
            r"\sum_i {{\lambda_i( \bm x, \bm v_i)}}\bm v_i {{=}} \sum_i {{( \bm b, \bm v_i)}} \bm v_i",
            color=BLUE,
        )
        problem_vector_replaced_4_a = MathTex(
            r"{{\lambda_i( \bm x, \bm v_i)}} {{=}} {{( \bm b, \bm v_i)}},\quad i=1,2",
            color=BLUE,
        )
        problem_vector_replaced_4_b = MathTex(
            r"{{( \bm x, \bm v_i)}}{{\lambda_i}} {{=}} {{( \bm b, \bm v_i)}}{{,\quad i=1,2}}",
            color=BLUE,
        )
        problem_vector_replaced_5 = MathTex(
            r"{{( \bm x, \bm v_i)}} {{=}} {{( \bm b, \bm v_i)}}/{{\lambda_i}}{{,\quad i=1,2}}",
            color=BLUE,
        )
        all_text_1st_phase.add(problem_vector)

        text_lin_trans = Tex("{{$D$}} is symmetric {{$(D=D^T)$}}")
        text_lin_trans.get_part_by_tex(r"$(D=D^T)$").set_color(BLUE)
        all_text_1st_phase.add(text_lin_trans)

        question_eigen = Tex("Ideally, what is the simplest effect $D$ could have?")
        eigen_problem_vector = MathTex(
            r"{{D \bm v = \lambda \bm v}} {{\to \begin{cases} \bm v_i,\\ \lambda_i, \end{cases} i=1,2}}",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)

        vec_in_phase_space = MathTex(
            r"\bm x = \sum_i {{( \bm x, \bm v_i)}} \bm v_i", color=BLUE
        )
        all_text_1st_phase.add(vec_in_phase_space)

        title.move_to(frame_u.get_center())
        all_text_1st_phase.arrange(DOWN, buff=std_line_buff)
        all_text_1st_phase.move_to(frame_text.get_center())

        diff_eq.move_to(problem_vector.get_center())
        diff_eq_simple.move_to(problem_vector.get_center())

        problem_vector_replaced_1_a.move_to(problem_vector.get_center())
        problem_vector_replaced_1_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_5_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_5_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_5_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_5_b.move_to(problem_vector.get_center())
        problem_vector_replaced_4_a.move_to(problem_vector.get_center())
        problem_vector_replaced_4_b.move_to(problem_vector.get_center())
        problem_vector_replaced_5.move_to(problem_vector.get_center())

        self.play(Write(title))
        self.wait(2)
        self.play(Write(diff_eq))
        self.wait(2)
        self.play(ReplacementTransform(diff_eq, diff_eq_simple))
        self.wait(2)
        self.play(ReplacementTransform(diff_eq_simple, problem_vector))
        self.wait(2)
        self.play(Write(text_lin_trans))
        self.wait(2)
        self.play(
            Write(eigen_problem_vector.get_part_by_tex(r"D \bm v = \lambda \bm v"))
        )
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"\to \begin{cases} \bm v_i,\\ \lambda_i, \end{cases} i=1,2"
                )
            )
        )
        self.wait(2)
        self.play(Write(vec_in_phase_space))
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
                text_lin_trans.get_part_by_tex(r"$(D=D^T)$"),
                color=ORANGE,
            ),
        )
        self.wait(2)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_b,
                problem_vector_replaced_2_5_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_2_5_a)
        self.add(problem_vector_replaced_2_5_b)
        self.play(
            Indicate(
                eigen_problem_vector.get_part_by_tex(r"D \bm v = \lambda \bm v"),
                color=ORANGE,
            ),
        )
        self.wait(1)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_5_b,
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
                problem_vector_replaced_3_5_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_3_5_a)
        self.add(problem_vector_replaced_3_5_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_3_5_b,
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
                problem_vector_replaced_5.get_part_by_tex(r"( \bm x, \bm v_i)"),
                color=ORANGE,
            ),
            Indicate(
                vec_in_phase_space.get_part_by_tex(r"( \bm x, \bm v_i)"),
                color=ORANGE,
            ),
        )

        self.wait(2)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    problem_vector_replaced_5,
                    vec_in_phase_space,
                    text_lin_trans,
                    title,
                    eigen_problem_vector,
                ]
            ]
        )
        self.wait(5)


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
        v_1 = MathTex(r"\bm v_1", color=ORANGE)
        v_2 = MathTex(r"\bm v_2", color=ORANGE)
        v_1.next_to(eig_vec_1, UP)
        v_2.next_to(eig_vec_2, RIGHT)

        self.play(DrawBorderThenFill(plane), run_time=2)
        self.wait(2)
        self.play(DrawBorderThenFill(eig_vec_1), DrawBorderThenFill(eig_vec_2))
        self.play(Write(v_1), Write(v_2))

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

        comp_1_tex = MathTex(r"( \bm x, \bm v_1)", color=BLUE)
        comp_2_tex = MathTex(r"( \bm x, \bm v_2)", color=BLUE)
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
        brace_comp_1 = BraceBetweenPoints(ORIGIN, comp_1 * eig_1, buff=0)
        line_comp_1 = DashedLine(random_vec_coord, random_vec_coord - comp_1 * eig_1)
        line_comp_1.stroke_width = 0.5
        line_comp_1.opacity = 0.5
        line_comp_2 = DashedLine(random_vec_coord, random_vec_coord - comp_2 * eig_2)
        brace_comp_2 = BraceBetweenPoints(comp_2 * eig_2, ORIGIN, buff=0)
        line_comp_2.stroke_width = 0.5
        line_comp_2.opacity = 0.5
        self.play(*[Unwrite(mobject) for mobject in [v_1, v_2]])
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
            run_time=std_wait_time,
        )
        comp_1_tex.next_to(brace_comp_2, LEFT)
        comp_2_tex.next_to(brace_comp_1, DOWN)
        self.play(*[Write(mobject) for mobject in [comp_1_tex, comp_2_tex]])
        comp_1_tex.add_background_rectangle()
        comp_2_tex.add_background_rectangle()

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


class FunctionsAsVectors(Scene):
    def construct(self):

        axes = Axes(
            x_range=[-2, 6, 1], x_length=7, y_range=[-5, 5, 1], y_length=5, tips=False
        )
        frame_text = Rectangle(height=6, width=7.0)
        frame_ur = Rectangle(height=2, width=7.0)
        frame_ul = Rectangle(height=2, width=7.0)
        up_frames = VGroup(frame_ul, frame_ur)
        up_frames.arrange(RIGHT, buff=0.7)
        frame_text.to_edge(RIGHT, buff=0.7)
        large_v_group = VGroup(axes, frame_text)
        large_v_group.arrange(RIGHT, buff=1)
        full_group = VGroup(up_frames, large_v_group)
        full_group.arrange(DOWN)
        full_group.center()
        # self.add(full_group)
        # self.add(frame_text)

        title = Tex(r"Vector/Function analogy").scale(1.5)
        title.move_to(frame_ur.get_center())

        axes_graph = axes.get_graph(
            lambda x: 4 * np.sin(np.pi / 3.5 * (x)), x_range=[-2, 6], color=ORANGE
        )
        vec_vals = [ValueTracker(0) for i in range(4)]

        changing_vec_tex = always_redraw(
            lambda: MathTex(
                "\\bm u = \\left\\{{\\begin{{array}}{{c}} {0:.2f} \\\\ {1:.2f} \\\\ {2:.2f} \\\\ {3:.2f} \\end{{array}}\\right\\}}".format(
                    *[val_tracker.get_value() for val_tracker in vec_vals]
                )
            )
            .move_to(frame_text.get_center())
            .set_color(BLUE)
        )
        vec_func_comp = MathTex(
            r"\bm u\colon \{1,2,3,4\}\to \mathbb R", color=BLUE
        ).move_to(frame_text.get_center())
        vec_func_comp_complex = MathTex(
            r"\bm u\colon \{1,2,3,4\}\to \mathbb C", color=BLUE
        ).move_to(frame_text.get_center())
        vec_func_int = MathTex(
            r"\bm u\colon \mathbb Z\to \mathbb R", color=BLUE
        ).move_to(frame_text.get_center())
        vec_func_real = MathTex(r"f\colon \mathbb R\to \mathbb R", color=BLUE).move_to(
            frame_text.get_center()
        )
        vec_func_complex = MathTex(
            r"f\colon \mathbb R\to \mathbb C", color=BLUE
        ).move_to(frame_text.get_center())
        inner_vec = MathTex(r"{{(\bm u, \bm v)=\sum_i u_i}} v_i", color=BLUE)
        inner_vec_comp = MathTex(
            r"{{(\bm u, \bm v)=\sum_i u_i}} \overline{ v_i }\,", color=BLUE
        )
        inner_func = MathTex(
            r"{{(f, g) = \int_I f(x)}}g(x){{\,\mathrm d x}}", color=BLUE
        )
        inner_func_comp = MathTex(
            r"{{(f, g) = \int_I f(x)}}\overline{g(x)}{{\,\mathrm d x }}\,",
            color=BLUE,
        )

        # Positioning
        # ==================================================================================
        all_phrases = VGroup(vec_func_comp, inner_vec, vec_func_int, inner_func)
        all_phrases.arrange(DOWN, buff=1)
        all_phrases.move_to(frame_text.get_center())

        vec_func_real.move_to(vec_func_int.get_center())
        vec_func_comp_complex.move_to(vec_func_comp.get_center())
        vec_func_complex.move_to(vec_func_real.get_center())
        inner_vec_comp.move_to(inner_vec.get_center())
        inner_func_comp.move_to(inner_func.get_center())

        y_0 = axes.coords_to_point(0, 0)[1]
        dots_vec = [
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(1, vec_vals[0].get_value()), color=ORANGE
                )
            ),
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(2, vec_vals[1].get_value()), color=ORANGE
                )
            ),
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(3, vec_vals[2].get_value()), color=ORANGE
                )
            ),
            always_redraw(
                lambda: Dot(
                    point=axes.coords_to_point(4, vec_vals[3].get_value()), color=ORANGE
                )
            ),
        ]
        dots_int_func = [
            Dot(
                point=axes.coords_to_point(i, 4 * np.sin(np.pi / 3.5 * (i))),
                color=ORANGE,
            )
            for i in [-2, -1, 0, 5, 6]
        ]
        lines_int_func = [
            DashedLine(
                axes.coords_to_point(i, 0),
                axes.coords_to_point(i, 4 * np.sin(np.pi / 3.5 * (i))),
            )
            for i in [-2, -1, 0, 5, 6]
        ]
        lines_vec = [
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(1, 0),
                    axes.coords_to_point(1, vec_vals[0].get_value()),
                )
            ),
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(2, 0),
                    axes.coords_to_point(2, vec_vals[1].get_value()),
                )
            ),
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(3, 0),
                    axes.coords_to_point(3, vec_vals[2].get_value()),
                )
            ),
            always_redraw(
                lambda: DashedLine(
                    axes.coords_to_point(4, 0),
                    axes.coords_to_point(4, vec_vals[3].get_value()),
                )
            ),
        ]

        # dots_vec = [
        #     always_redraw(
        #         lambda: Dot(point=axes.coords_to_point(1, vec_vals[0])),
        #     ),
        #     always_redraw(
        #         lambda: Dot(point=axes.coords_to_point(2, vec_vals[1])),
        #     ),
        # ]

        self.play(Write(title))
        self.wait(1)
        self.play(Create(axes))
        self.play(Write(changing_vec_tex))
        self.play(
            *[DrawBorderThenFill(line) for line in lines_vec],
            *[DrawBorderThenFill(dot) for dot in dots_vec],
        )
        np.random.seed(42)
        # self.play(
        #     *[
        #         value_tracker.animate.set_value(8 * np.random.random() - 4)
        #         for i, value_tracker in enumerate(vec_vals)
        #     ],
        #     rate_func=linear,
        #     run_time=4,
        # )
        self.play(
            *[
                value_tracker.animate.set_value(8 * np.random.random() - 4)
                for i, value_tracker in enumerate(vec_vals)
            ],
            rate_func=linear,
            run_time=4,
        )
        self.play(
            *[
                value_tracker.animate.set_value(4 * np.sin(np.pi / 3.5 * (i + 1)))
                for i, value_tracker in enumerate(vec_vals)
            ],
            rate_func=linear,
            run_time=4,
        )
        self.play(ReplacementTransform(changing_vec_tex, vec_func_comp))
        self.wait(2)
        self.play(Write(inner_vec))
        self.wait(1)
        self.play(
            *[DrawBorderThenFill(line) for line in lines_int_func],
            *[DrawBorderThenFill(dot) for dot in dots_int_func],
        )
        self.play(TransformFromCopy(vec_func_comp, vec_func_int))
        self.wait(1)
        self.play(Create(axes_graph))
        self.play(ReplacementTransform(vec_func_int, vec_func_real))
        self.wait(1)
        self.play(TransformFromCopy(inner_vec, inner_func))
        self.wait(1)
        self.play(
            ReplacementTransform(vec_func_comp, vec_func_comp_complex),
            ReplacementTransform(vec_func_real, vec_func_complex),
        )
        self.wait(1)
        self.play(
            ReplacementTransform(inner_vec, inner_vec_comp),
            ReplacementTransform(inner_func, inner_func_comp),
        )
        self.wait(5)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    inner_vec_comp,
                    inner_func_comp,
                    title,
                    vec_func_comp_complex,
                    vec_func_complex,
                ]
            ],
            Uncreate(axes),
            Uncreate(axes_graph),
            *[
                Uncreate(mobject)
                for mobject in dots_vec + dots_int_func + lines_vec + lines_int_func
            ],
        )


class FourierSeriesScene(Scene):
    def construct(self):

        big_frame = Rectangle(height=6.0, width=14.0)
        frame_u = Rectangle(height=2, width=14.0)
        full_group = VGroup(frame_u, big_frame)
        full_group.arrange(DOWN)
        full_group.center()

        title = Tex("Periodic functions").scale(1.5)
        title.to_edge(UP, buff=1)
        self.add(title)
        #
        all_text_1st_phase = VGroup()
        #
        problem_vector = MathTex(r"{{Df}} {{=}} {{g}}", color=BLUE)
        problem_vector_replaced_1_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty} }}( {{Df}}, \varphi_n) \varphi_n(x) {{=}} \sum_{n=-\infty}^{+\infty} ( {{g}}, \varphi_n) \varphi_n(x)",
            color=BLUE,
        )
        problem_vector_replaced_1_b = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty} (}} {{D}}{{f}},{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty} (}} {{f}}, {{D}}^\dagger{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_b = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, }}{D}^\dagger{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_5_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, }}-{{D}}{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_2_5_b = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, -}}D{{\varphi_n)}} {{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_3_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{( f, -}}\frac{2\pi in}{L}{{\varphi_n)}}{{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_3_b = MathTex(
            r"\sum_{n=-\infty}^{+\infty} {{( f,}} {{-\frac{2\pi in}{L} }} {{\varphi_n)}}\varphi_n(x) {{=}} \sum_{n=-\infty}^{+\infty} {{( g, \varphi_n)}} \varphi_n(x)",
            color=BLUE,
        )
        problem_vector_replaced_3_5_a = MathTex(
            r"{{\sum_{n=-\infty}^{+\infty}}} {{-\frac{2\pi inx}{L} }}{{( f,}} {{\varphi_n)}}{{\varphi_n(x)}} {{=}} {{\sum_{n=-\infty}^{+\infty} ( g, \varphi_n) \varphi_n(x)}}",
            color=BLUE,
        )
        problem_vector_replaced_3_5_b = MathTex(
            r"\sum_{n=-\infty}^{+\infty} {{-\frac{2\pi inx}{L} }}{{( f, \varphi_n)}}\varphi_n(x) {{=}} \sum_{n=-\infty}^{+\infty} {{( g, \varphi_n)}} \varphi_n(x)",
            color=BLUE,
        )
        problem_vector_replaced_4_a = MathTex(
            r"{{-\frac{2\pi inx}{L} }}{{( f, \varphi_n)}} {{=}} {{( g, \varphi_n)}},\quad n\in\mathbb Z",
            color=BLUE,
        )
        problem_vector_replaced_4_b = MathTex(
            r"{{-\frac{2\pi inx}{L} }}{{( f, \varphi_n)}} {{=}} {{( g, \varphi_n)}}{{,\quad n\in\mathbb Z}}",
            color=BLUE,
        )
        problem_vector_replaced_5 = MathTex(
            r"{{( f, \varphi_n)}} {{=}} {{i}}{{( g, \varphi_n)}}/{{\frac{L}{2\pi n x} }}{{,\quad n\in\mathbb Z}}",
            color=BLUE,
        )
        all_text_1st_phase.add(problem_vector)

        text_lin_trans = Tex("$D$ is anti-Hermitian {{($D^\dagger = -D$)}}")
        text_lin_trans.get_part_by_tex(r"($D^\dagger = -D$)").set_color(BLUE)
        all_text_1st_phase.add(text_lin_trans)

        question_eigen = Tex("Ideally, what is the simplest effect $D$ could have?")
        eigen_problem_vector = MathTex(
            r"{{-iD \varphi(x) = \lambda \varphi(x)}} {{\to \varphi_n(x) = \frac{e^\frac{2\pi inx}{L} }{\sqrt{2\pi} },\quad n \in \mathbb Z}}",
            color=BLUE,
        )
        eigen_problem_vector_2 = MathTex(
            r"D \varphi(x) = i\lambda \varphi(x)",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)

        vec_in_phase_space_a = MathTex(
            r"{{f(x) = \sum_{n=-\infty}^{+\infty} }}{{\int_{-L/2}^{L/2} \frac{e^\frac{-2\pi i n y}{L} }{\sqrt{2\pi} }\mathrm d y}} \frac{e^\frac{2\pi i n x}{L} }{\sqrt{2\pi} }",
            color=BLUE,
        )
        vec_in_phase_space = MathTex(
            r"{{f(x) = \sum_{n=-\infty}^{+\infty} }} {{(f, \varphi_n)}} \varphi_n(x)",
            color=BLUE,
        )
        all_text_1st_phase.add(vec_in_phase_space)

        all_text_1st_phase.arrange(DOWN, buff=std_line_buff)
        all_text_1st_phase.move_to(big_frame.get_center())
        braces = Brace(vec_in_phase_space, RIGHT)
        eq_text = braces.get_text("Fourier series")

        problem_vector_replaced_1_a.move_to(problem_vector.get_center())
        problem_vector_replaced_1_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_5_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_5_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_5_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_5_b.move_to(problem_vector.get_center())
        problem_vector_replaced_4_a.move_to(problem_vector.get_center())
        problem_vector_replaced_4_b.move_to(problem_vector.get_center())
        problem_vector_replaced_5.move_to(problem_vector.get_center())

        eigen_problem_vector_2.move_to(
            eigen_problem_vector.get_part_by_tex(r"-iD \varphi(x) = \lambda \varphi(x)")
        )

        vec_in_phase_space_a.move_to(vec_in_phase_space)

        self.play(Write(title))
        self.wait(2)
        self.add(problem_vector)
        self.wait(2)
        self.play(Write(text_lin_trans))
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"-iD \varphi(x) = \lambda \varphi(x)"
                )
            )
        )
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"\to \varphi_n(x) = \frac{e^\frac{2\pi inx}{L} }{\sqrt{2\pi} },\quad n \in \mathbb Z"
                )
            )
        )
        self.wait(2)
        self.play(Write(vec_in_phase_space_a))
        self.wait(2)
        self.play(ReplacementTransform(vec_in_phase_space_a, vec_in_phase_space))

        self.wait(2)
        self.play(GrowFromCenter(braces), Write(eq_text))
        self.wait(1)
        self.play(Indicate(vec_in_phase_space, color=ORANGE))
        self.wait(2)
        self.play(
            # ReplacementTransform(problem_vector.get_part_by_tex(r"f"), test),
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
        self.remove(problem_vector_replaced_2_a)
        self.add(problem_vector_replaced_2_b)
        self.wait(2)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_b,
                problem_vector_replaced_2_5_a,
                run_time=std_wait_time,
            )
        )
        self.remove(problem_vector_replaced_2_5_a)
        self.add(problem_vector_replaced_2_5_b)
        self.wait(2)

        self.play(
            Indicate(
                eigen_problem_vector.get_part_by_tex(
                    r"-iD \varphi(x) = \lambda \varphi(x)"
                ),
                color=ORANGE,
            ),
        )
        self.wait(1)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_2_5_b,
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
                problem_vector_replaced_3_5_a,
                run_time=std_wait_time,
            )
        )
        self.wait(2)
        self.remove(problem_vector_replaced_3_5_a)
        self.add(problem_vector_replaced_3_5_b)
        self.play(
            TransformMatchingTex(
                problem_vector_replaced_3_5_b,
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
                problem_vector_replaced_5.get_part_by_tex(r"( f, \varphi_n)"),
                color=ORANGE,
            ),
            Indicate(
                vec_in_phase_space.get_part_by_tex(r"(f, \varphi_n)"),
                color=ORANGE,
            ),
        )

        self.wait(2)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    title,
                    problem_vector_replaced_5,
                    text_lin_trans,
                    eigen_problem_vector,
                    vec_in_phase_space,
                    braces,
                    eq_text,
                ]
            ]
        )

        self.wait(10)


class FourierIntegralScene(Scene):
    def construct(self):

        big_frame = Rectangle(height=6.0, width=14.0)
        frame_u = Rectangle(height=2, width=14.0)
        full_group = VGroup(frame_u, big_frame)
        full_group.arrange(DOWN)
        full_group.center()

        # self.add(big_frame)
        # self.add(frames)

        title = Tex("Functions").scale(1.5)
        title.move_to(frame_u.get_center())
        #
        all_text_1st_phase = VGroup()
        #
        problem_vector = MathTex(r"{{D}} {{f}} {{=}} {{g}}", color=BLUE)
        problem_vector_replaced_1_a = MathTex(
            r"\int_{-\infty}^{+\infty} ({{D}}{{f}}, \varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w {{=}} \int_{-\infty}^{+\infty} ({{g}}, \varphi) \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w",
            color=BLUE,
        )
        problem_vector_replaced_1_b = MathTex(
            r"{{\int_{-\infty}^{+\infty} }} {{({{D}}f,\varphi)}} {{\frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_2_a = MathTex(
            r"{{\int_{-\infty}^{+\infty} }} {{(f,}}-{{D}}{{\varphi)}} {{\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_2_b = MathTex(
            r"{{\int_{-\infty}^{+\infty}(f,-D\varphi)}} {{\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_3_a = MathTex(
            r"{{\int_{-\infty}^{+\infty}(f,-iw\varphi)}}{{\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w}} {{=}} {{\int_{-\infty}^{+\infty} (g,\varphi) \frac{e^{ixw} }{\sqrt{2\pi} } \mathrm d w}}",
            color=BLUE,
        )
        problem_vector_replaced_3_b = MathTex(
            r"\int_{-\infty}^{+\infty} {{(f,-iw\varphi)}}\frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w {{=}} \int_{-\infty}^{+\infty} {{(g,\varphi)}} \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w",
            color=BLUE,
        )
        problem_vector_replaced_4_a = MathTex(
            r"{{-iw(f,\varphi)}} {{=}} {{(g,\varphi)}},\quad w\in \mathbb R",
            color=BLUE,
        )
        problem_vector_replaced_4_b = MathTex(
            r"-{{i}}{{w}}{{(f,\varphi)}} {{=}} {{(g,\varphi)}}{{,\quad w\in \mathbb R}}",
            color=BLUE,
        )
        problem_vector_replaced_5 = MathTex(
            r"{{(f,\varphi)}} {{=}} {{i}}{{(g,\varphi)}}/{{w}}{{,\quad w\in \mathbb R}}",
            color=BLUE,
        )
        problem_vector_replaced_6 = MathTex(
            r"{{[\mathcal F(f)](w)}} {{=}} {{i}}{{[\mathcal F(g)](w)}}/{{w}}{{,\quad w\in \mathbb R}}",
            color=BLUE,
        )
        all_text_1st_phase.add(problem_vector)

        text_lin_trans = Tex("$D$ is anti-Hermitian {{($D^\dagger = -D$)}}")
        text_lin_trans.get_part_by_tex(r"($D^\dagger = -D$)").set_color(BLUE)
        all_text_1st_phase.add(text_lin_trans)

        question_eigen = Tex("Ideally, what is the simplest effect $D$ could have?")
        eigen_problem_vector = MathTex(
            r"{{-iD \varphi(x) = w \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        eigen_problem_vector_1_b = MathTex(
            r"-{{i}}{{D \varphi(x) =}} {{w \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)
        eigen_problem_vector_2_a = MathTex(
            r"{{D \varphi(x) =}} i{{w \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        eigen_problem_vector_2_b = MathTex(
            r"{{D \varphi(x) = iw \varphi(x)}} {{\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R}}",
            color=BLUE,
        )
        all_text_1st_phase.add(eigen_problem_vector)

        vec_in_phase_space = MathTex(
            r"f(x) = \int_{-\infty}^{+\infty} {{[\mathcal F(f)](w)}} \frac{e^{ixw} }{\sqrt{2\pi} }\mathrm d w",
            color=BLUE,
        )
        all_text_1st_phase.add(vec_in_phase_space)
        vec_in_phase_space_2 = MathTex(
            r"f(x) = \int_{-\infty}^{+\infty} {{(f, \varphi)}} \varphi(w)\mathrm d w",
            color=BLUE,
        )

        all_text_1st_phase.arrange(DOWN, buff=std_line_buff)
        all_text_1st_phase.move_to(big_frame.get_center())

        problem_vector_replaced_1_a.move_to(problem_vector.get_center())
        problem_vector_replaced_1_b.move_to(problem_vector.get_center())
        problem_vector_replaced_2_a.move_to(problem_vector.get_center())
        problem_vector_replaced_2_b.move_to(problem_vector.get_center())
        problem_vector_replaced_3_a.move_to(problem_vector.get_center())
        problem_vector_replaced_3_b.move_to(problem_vector.get_center())
        problem_vector_replaced_4_a.move_to(problem_vector.get_center())
        problem_vector_replaced_4_b.move_to(problem_vector.get_center())
        problem_vector_replaced_5.move_to(problem_vector.get_center())
        problem_vector_replaced_6.move_to(problem_vector.get_center())

        vec_in_phase_space_2.move_to(vec_in_phase_space.get_center())

        eigen_problem_vector_1_b.move_to(eigen_problem_vector)
        eigen_problem_vector_2_a.move_to(eigen_problem_vector)
        eigen_problem_vector_2_b.move_to(eigen_problem_vector)

        braces = Brace(vec_in_phase_space_2, RIGHT)
        eq_text = braces.get_text("Fourier integral")

        self.play(Write(title))
        self.wait(2)
        self.add(problem_vector)
        self.wait(2)
        self.play(Write(text_lin_trans))
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(r"-iD \varphi(x) = w \varphi(x)")
            )
        )
        self.wait(2)
        self.play(
            Write(
                eigen_problem_vector.get_part_by_tex(
                    r"\to \varphi(x) = \frac{e^{iwx} }{\sqrt{2\pi} },\quad w \in \mathbb R"
                )
            )
        )
        self.wait(2)
        self.play(Write(vec_in_phase_space))
        self.wait(2)
        self.play(
            ReplacementTransform(vec_in_phase_space, vec_in_phase_space_2), run_time=2
        )
        self.remove(vec_in_phase_space)
        self.add(vec_in_phase_space_2)

        self.play(GrowFromCenter(braces), Write(eq_text))

        self.wait(2)
        self.play(Indicate(vec_in_phase_space_2, color=ORANGE))
        self.wait(1)
        self.play(
            # ReplacementTransform(problem_vector.get_part_by_tex(r"f"), test),
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
                eigen_problem_vector.get_part_by_tex(r"-iD \varphi(x) = w \varphi(x)"),
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
            ReplacementTransform(problem_vector_replaced_5, problem_vector_replaced_6)
        )

        self.wait(2)
        self.play(
            Indicate(
                problem_vector_replaced_6.get_part_by_tex(r"[\mathcal F(f)](w)"),
                color=ORANGE,
            ),
            Indicate(
                vec_in_phase_space_2.get_part_by_tex(r"(f, \varphi)"),
                color=ORANGE,
            ),
        )
        self.wait(2)
        self.play(
            *[
                Unwrite(mobject)
                for mobject in [
                    title,
                    problem_vector_replaced_6,
                    text_lin_trans,
                    eigen_problem_vector,
                    vec_in_phase_space_2,
                    braces,
                    eq_text,
                ]
            ]
        )

        self.wait(10)


class SplitScreenDerivativeMultiplication(Scene):
    def construct(self):

        # Function graph setup
        # ==================================================================================
        axes_func = Axes(
            x_range=[-6, 6],
            y_range=[-2, 2],
            x_length=5,
            y_length=4,
            tips=False,
            x_axis_config={"include_ticks": False},
            y_axis_config={"include_ticks": False},
        )
        box_axes = SurroundingRectangle(axes_func, buff=0)
        magnitude = ValueTracker(1.0)
        theta = ValueTracker(0.0)
        omega = 3

        sin_func_og = axes_func.get_graph(
            lambda t: np.real(np.exp(1j * omega * t)), stroke_opacity=0.5
        )
        sin_func = always_redraw(
            lambda: axes_func.get_graph(
                lambda t: np.real(
                    magnitude.get_value()
                    * np.exp(1j * theta.get_value())
                    * np.exp(1j * omega * t)
                ),
                x_range=[-6, 6],
            )
        )
        full_plot = VGroup(sin_func, box_axes, axes_func, sin_func_og)

        # Complex plane setup
        # ==================================================================================

        plane = ComplexPlane(
            x_range=[-2, 2],
            y_range=[-2, 2],
            x_length=5,
            y_length=4,
        )
        # plane_polar = ComplexPlane(
        #     x_range=[-2, 2],
        #     y_range=[-2, 2],
        #     x_length=5,
        #     y_length=4,
        #     axis_config={"include_ticks": False, "numbers_to_exclude": [-2, 2]},
        # )

        box_plane = SurroundingRectangle(plane, buff=0)

        full_plane = VGroup(plane, box_plane)  # , labels)

        everything = VGroup(full_plot, full_plane)
        everything.arrange(RIGHT, buff=1)
        everything.center()

        labels = plane.get_axis_labels(x_label="\mathrm{Re}", y_label="\mathrm{Im}")
        labels[1].shift(0.5 * DOWN)
        labels[0].shift(0.7 * LEFT)

        self.add(full_plane, labels)
        d1 = always_redraw(
            lambda: Dot(
                plane.n2p(magnitude.get_value() * np.exp(1j * theta.get_value())),
                color=YELLOW,
            )
        )
        self.add(
            d1,
        )
        np.random.seed(40)
        magnitude_random_values = 1.5 * np.random.rand(4)
        theta_random_values = 2 * np.pi * np.random.rand(4)

        # Animation
        # ==================================================================================

        self.add(sin_func, box_axes, sin_func_og)
        self.play(
            magnitude.animate.set_value(1.4), run_time=2, rate_func=there_and_back
        )
        self.wait(1)
        self.play(theta.animate.set_value(5), run_time=4, rate_func=there_and_back)
        self.wait(1)
        for i in range(4):
            self.play(
                magnitude.animate.set_value(magnitude_random_values[i]),
                theta.animate.set_value(theta_random_values[i]),
                run_time=3,
                rate_func=smooth,
            )
        self.play(
            magnitude.animate.set_value(1.2),
            theta.animate.set_value(np.pi / 3),
            run_time=3,
            rate_func=smooth,
        )
        self.wait()


class FourierSeriesExplainer(ThreeDScene):
    def func_to_model(self, x, period=1):

        k = np.round(x / period)
        x = x - period * k

        # amplitude = 1
        # val_saw = -amplitude + (x + period / 2) ** 2 * (2 * amplitude) / period
        # val_square = amplitude / 2 * x ** 3 if -period / 4 <= x <= period / 4 else 0

        amplitude = 1
        val = amplitude * np.sin(2 * np.pi * x / period) if 0 <= x <= period / 2 else 0

        return val

    def construct(self):

        axes_func = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-4, 4, 1],
            x_length=8,
            y_length=6,
            z_length=6,
        )
        axes_coeffs = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-2.5 * 4, 2.5 * 4, 1],
            x_length=4,
            y_length=6,
            z_length=2.5 * 6,
            axis_config={"include_ticks": False},
            tips=False,
        )

        graph = axes_func.get_graph(
            self.func_to_model, x_range=[-1.5, 1.5, 1e-3], color=YELLOW
        )

        # rects = axes_func.get_riemann_rectangles(
        #     graph=graph, x_range=[-2, 2], dx=0.1, stroke_color=WHITE
        # )
        #
        # graph2 = axes_func.get_parametric_curve(
        #     lambda t: np.array([np.cos(t), np.sin(t), t]),
        #     t_range=[-2 * PI, 2 * PI],
        #     color=RED,
        # )
        all_func = VGroup(axes_func, graph)
        all_coeff = VGroup(axes_coeffs)
        both_graphs = VGroup(all_func, all_coeff)
        both_graphs.arrange(RIGHT)
        both_graphs.center()
        self.add(graph, all_coeff)
        basis_func = []
        delta_z = 2.5
        n_points = 500
        t = np.linspace(-1.5, 1.5, n_points)
        for freq in range(-3, 4):
            x_values = t
            y_values = np.real(np.exp(1j * freq * t * 2 * np.pi / 1))
            z_values = n_points * [delta_z * freq]

            basis_func.append(
                axes_func.get_line_graph(
                    x_values=x_values,
                    y_values=y_values,
                    z_values=z_values,
                    line_color=BLUE,
                    add_vertex_dots=False,
                )
            )
            if freq > 0:
                basis_func[-1].set_stroke(color=GREEN, opacity=0.5, width=0.5)
            else:
                basis_func[-1].set_stroke(color=GREEN)
        self.add(*basis_func)

        dashed_lines = []
        t = np.linspace(-2, 5, 10)
        for freq in range(-3, 4):
            x_values = t
            y_values = 10 * [0]
            z_values = 10 * [delta_z * freq]

            dashed_lines.append(
                Line(
                    axes_func.coords_to_point(*[x_values[0], y_values[0], z_values[0]]),
                    axes_func.coords_to_point(
                        *[x_values[-1], y_values[-1], z_values[-1]]
                    ),
                )
            )
            if freq > 0:
                dashed_lines[-1].set_stroke(color=WHITE, opacity=0.5, width=0.5)
            else:
                dashed_lines[-1].set_stroke(color=WHITE, opacity=0.8, width=0.3)
        self.add(*dashed_lines)
        # self.wait()

        # The camera is auto set to PHI = 0 and THETA = -90
        coeff_dots = []
        magnitude_trackers = []
        theta_trackers = []
        for freq in range(-3, 4):
            magnitude_trackers.append(ValueTracker(1))
            theta_trackers.append(ValueTracker(0))

            coeff_dots.append(
                always_redraw(
                    lambda: Dot(
                        axes_coeffs.coords_to_point(
                            np.real(
                                magnitude_trackers[-1].get_value()
                                * np.exp(1j * theta_trackers[-1].get_value())
                            ),
                            np.imag(
                                magnitude_trackers[-1].get_value()
                                * np.exp(1j * theta_trackers[-1].get_value())
                            ),
                            delta_z * freq,
                        )
                    )
                )
            )
            # if freq > 0:
            #     coeff_dots[-1].set_stroke(color=WHITE, opacity=0.5, width=0.5)
            # else:
            #     coeff_dots[-1].set_stroke(color=WHITE, opacity=0.8, width=0.3)
        self.add(*coeff_dots)

        theta_prime = 45 * DEGREES
        phi_prime = 10 * DEGREES

        position_prime = [
            np.sin(phi_prime) * np.cos(theta_prime),
            np.sin(phi_prime) * np.sin(theta_prime),
            np.cos(phi_prime),
        ]
        position_prime = [2, 1.5, 1]
        up = np.array([0, 1, 0])
        camera_direction = np.array(position_prime)
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        camera_right = np.cross(camera_direction, up)
        camera_right = camera_right / np.linalg.norm(camera_right)
        camera_up = np.cross(camera_direction, camera_right)
        camera_up = camera_up / np.linalg.norm(camera_up)

        # theta_cam = np.arccos(position_prime[2] / np.linalg.norm(position_prime))
        # phi_cam = np.arctan(position_prime[1] / position_prime[0])
        #
        # view_up = [0.183013, 0.965926, 0.183013]
        # gamma_cam = np.arccos(
        #     np.dot(position_prime, view_up)
        #     / (np.linalg.norm(position_prime) * np.linalg.norm(view_up))
        # )
        # sdfa
        rot_mat = np.array([camera_right, camera_up, camera_direction])
        rot_mat_inv = rot_mat
        phi_cam = np.arctan2(rot_mat_inv[2, 0], rot_mat_inv[2, 1])
        # phi_cam = np.arccos(
        #     -camera_direction[1] / np.sqrt(1 - camera_direction[2] ** 2)
        # )
        theta_cam = np.arccos(rot_mat_inv[2, 2])
        gamma_cam = -np.arctan2(rot_mat_inv[0, 2], rot_mat_inv[1, 2])
        # gamma_cam = np.arccos(camera_right[2] / np.sqrt(1 - camera_direction[2] ** 2))
        print(phi_cam / DEGREES, theta_cam / DEGREES, gamma_cam / DEGREES)
        # self.set_camera_orientation(
        self.wait(5)
        self.move_camera(
            # phi=20 * DEGREES,
            #     theta=46 * DEGREES - 90 * DEGREES,
            #     gamma=-15 * DEGREES,
            phi=-phi_cam,
            theta=-theta_cam - 90 * DEGREES,
            gamma=gamma_cam,
        )
        #  array([[-7.07106781e-01,  1.52177069e-16,  7.07106781e-01],
        # [ 4.08248290e-01, -8.16496581e-01,  4.08248290e-01],
        # [ 5.77350269e-01,  5.77350269e-01,  5.77350269e-01]])

        # self.wait(10)
        # self.move_camera(theta=-70 * DEGREES)
        # self.wait(1)

        # self.begin_ambient_camera_rotation(
        #     rate=PI / 10, about="gamma"
        # )  # Rotates at a rate of radians per second
        self.wait(10)
        # self.play(Create(rects), run_time=3)
        # self.play(Create(graph2))
        # self.wait()
        # self.stop_ambient_camera_rotation()
        #
        # self.wait()
        # self.begin_ambient_camera_rotation(
        #     rate=PI / 10, about="phi"
        # )  # Rotates at a rate of radians per second
        # self.wait(2)
        # self.stop_ambient_camera_rotation()


class Intro(Scene):
    def construct(self):
        text_1 = Tex(r"Goal: A more intuitive understanding of the Fourier tranform:")
        fourier_transform = MathTex(
            r"{{[\mathcal F(f)](w)}}=\int_{-\infty}^{+\infty} f(t) e^{-iwt}\mathrm d t",
            color=BLUE,
        )
        text_2 = Tex(
            r"Why are a differential equations transformed into algebraic equations by the Fourier transform?"
        )
        diff_eq = MathTex(r"m{{D^2}} f(t) + c {{D}}f(t) + kf(t) = F(t)", color=BLUE)
        diff_eq_2 = MathTex(
            r"m{{\frac{\mathrm d^2}{\mathrm d x^2}}} f(t) + c {{\frac{\mathrm d}{\mathrm dx} }}f(t) + kf(t) = F(t)",
            color=BLUE,
        )
        alg_eq = MathTex(
            r"-mw^2{{[\mathcal F(f)](w)}}+icw{{[\mathcal F(f)](w)}} + k{{[\mathcal F(f)](w)}}={{[\mathcal F(F)](w)}}",
            color=BLUE,
        )
        sol = MathTex(
            r"{{f(t)}} =\frac{1}{2\pi}\int_{-\infty}^{+\infty}{{[\mathcal F(f)](w)}}e^{iwt}\mathrm d w",
            color=BLUE,
        )
        text_3 = Tex(
            r"Why the use of {{$e^{-iwt}$}} as the kernel of the tranformation? How can this be interpreted?"
        )
        text_3.get_part_by_tex(r"$e^{-iwt}$").set_color(BLUE)
        full_group = VGroup(*[text_1, fourier_transform, text_2, diff_eq, text_3])

        full_group.arrange(DOWN, buff=std_line_buff)
        full_group.center()
        alg_eq.move_to(diff_eq.get_center())
        diff_eq_2.move_to(diff_eq.get_center())
        sol.move_to(diff_eq_2.get_center())

        self.play(Write(text_1), Write(fourier_transform), run_time=2)
        self.wait(1)
        self.play(Indicate(fourier_transform, color=ORANGE))
        self.wait(1)
        self.play(Write(text_2), Write(diff_eq), run_time=2)
        self.wait(1)
        self.play(
            ReplacementTransform(diff_eq, diff_eq_2),
            run_time=2,
        )
        self.wait(1)
        self.play(ReplacementTransform(diff_eq_2, alg_eq), run_time=2)
        self.wait(1)
        self.play(
            Indicate(alg_eq.get_parts_by_tex(r"[\mathcal F(f)](w)"), color=ORANGE)
        )
        self.wait(1)
        self.play(ReplacementTransform(alg_eq, sol), run_time=2)
        self.wait(1)
        self.play(Indicate(sol.get_part_by_tex(r"f(t)"), color=ORANGE))
        self.wait(1)
        self.play(Write(text_3), run_time=2)
        self.wait(2)

        self.play(
            *[
                Unwrite(mobject)
                for mobject in [text_1, fourier_transform, text_2, sol, text_3]
            ]
        )
        self.wait(2)

        # text_5 = Tex(r"We are going the analogy between ordinary vectors and functions as a starting point.")


class Outro(Scene):
    def construct(self):
        text_2 = Tex(
            r"Why are a differential equations transformed into algebraic equations by the Fourier transform?"
        )
        text_3 = Tex(
            r"Why the use of {{$e^{-iwt}$}} as the kernel of the tranformation? How can this be interpreted?"
        )
        text_3.get_part_by_tex(r"$e^{-iwt}$").set_color(BLUE)
        full_group = VGroup(*[text_2, text_3])

        full_group.arrange(DOWN, buff=3)
        full_group.center()

        self.play(Write(text_2), Write(text_3), run_time=2)
        self.wait(2)

        self.play(*[Unwrite(mobject) for mobject in [text_2, text_3]])
        self.wait(2)

        # text_5 = Tex(r"We are going the analogy between ordinary vectors and functions as a starting point.")


def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag
